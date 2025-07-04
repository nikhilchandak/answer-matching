import os
import json
import glob
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict

# Import utility functions
from utils import (
    load_jsonl_file,
    convert_rating_to_binary,
    get_balanced_question_ids,
    plot_alignment,
    plot_accuracy,
    plot_accuracy_and_alignment,
    analyze_error_types,
    select_questions_and_calculate_weights,
    calculate_agreement_metric,
    collect_scotts_pi_data
)

class AlignmentAnalyzer:
    def __init__(self, 
                 human_annotations_dir: str,
                 mcq_file: str,
                 lm_matchings_file: str,
                 unique_rating_filter: Tuple[int, int] = (1, 5),
                 specific_filter: Tuple[int, int] = (1, 5),
                 exclude_matchers: List[str] = None,
                 dataset_name: str = None,
                 cloze_file: str = None,
                 verification_file: str = None):
        """
        Initialize the AlignmentAnalyzer with file paths.
        
        Args:
            human_annotations_dir: Directory containing human annotation files
            mcq_file: File containing model MCQ responses
            lm_matchings_file: File containing model free-form responses and LM matchings
            unique_rating_filter: Range for filtering by rating_multians (min, max)
            specific_filter: Range for filtering by rating_osq (min, max)
            exclude_matchers: List of matcher models to exclude from analysis
            dataset_name: Name of the dataset (for special handling, e.g., 'math')
            cloze_file: File containing MC Cloze responses
            verification_file: File containing MC Verify responses
        """
        self.human_annotations_dir = human_annotations_dir
        self.mcq_file = mcq_file
        self.lm_matchings_file = lm_matchings_file
        self.unique_rating_filter = unique_rating_filter
        self.specific_filter = specific_filter
        self.exclude_matchers = exclude_matchers or ["qwen-2.5-14b-instruct"]
        # self.exclude_matchers = ["qwen-2.5-7b-instruct-JUDGE"]
        self.dataset_name = dataset_name
        self.is_math = (dataset_name == "math")
        self.is_gpqa_diamond = (dataset_name == "gpqa_diamond")
        
        # Initialize data containers
        self.human_annotators = {}  # Will store all human annotations by annotator ID
        self.ground_truth = {}      # Will store the ground truth annotations
        self.mcq_responses = {}     # Will store MCQ responses
        self.lm_matchings = {}      # Will store LM matchings
        self.matchers = set()       # Will store all matcher names
        self.filtered_question_ids = set()  # Will store filtered question IDs
        self.cloze_file = cloze_file
        self.verification_file = verification_file
        self.cloze_responses = {}   # Will store MC Cloze responses
        self.verify_responses = {}  # Will store MC Verify responses
        
        # Load all data
        if not self.is_math:
            self.load_human_annotations()
        self.load_mcq_responses()
        self.load_lm_matchings()
        self.load_verify_responses()
        self.load_cloze_responses()

    def load_human_annotations(self) -> None:
        """Load all human annotation files from the specified directory."""
        # Get all JSONL files in the directory
        annotation_files = glob.glob(os.path.join(self.human_annotations_dir, "*.jsonl"))
        print(f"Found {len(annotation_files)} human annotation files: {', '.join([os.path.basename(file) for file in annotation_files])}")
        
        # Process each file
        for i, file_path in enumerate(annotation_files):
            annotator_id = f"human_{i+1}"
            self.human_annotators[annotator_id] = self._load_human_annotation_file(file_path)
            print(f"Loaded {len(self.human_annotators[annotator_id])} annotations from {annotator_id}")
    
    def _load_human_annotation_file(self, file_path: str) -> Dict:
        annotations = {}
        data = load_jsonl_file(file_path)
        if self.is_gpqa_diamond:
            for item in data:
                question_id = str(item.get("question_id"))
                models = item.get("model", [])
                rating_matches = item.get("rating_match", [])
                if not isinstance(models, list) or not isinstance(rating_matches, list):
                    continue
                for i, model in enumerate(models):
                    if i < len(rating_matches):
                        entry = dict(item)
                        entry["rating_match"] = rating_matches[i]
                        entry["model"] = model
                        annotations[(question_id, model)] = entry
        else:
            for item in data:
                question_id = str(item.get("question_id"))
                if question_id is not None:
                    annotations[question_id] = item
        return annotations
    
    def define_ground_truth(self, annotator_idx: int = 0) -> str:
        """
        Define ground truth based on a specific human annotator, or for math dataset, from lm_matchings_file.
        
        Args:
            annotator_idx: Index of the human annotator to use as ground truth (0-based)
            
        Returns:
            String identifier of the ground truth annotator
        """
        if self.is_math:
            # For math, use exact_match from lm_matchings_file as ground truth
            lm_data = load_jsonl_file(self.lm_matchings_file)
            self.ground_truth = {}
            for item in lm_data:
                question_id = str(item.get("question_id"))
                exact_match = item.get("exact_match")
                if question_id is not None and exact_match is not None:
                    # Accept 0/1 as int or str
                    if isinstance(exact_match, str) and exact_match in ["0", "1"]:
                        self.ground_truth[question_id] = {"rating_match": int(exact_match)}
                    elif isinstance(exact_match, int) and exact_match in [0, 1]:
                        self.ground_truth[question_id] = {"rating_match": exact_match}
            self.filtered_question_ids = set(self.ground_truth.keys())
            print(f"Loaded {len(self.ground_truth)} math ground truth samples from lm_matchings_file.")
            return "0"
        # Original behavior for non-math datasets
        annotator_keys = list(self.human_annotators.keys())
        if annotator_idx >= len(annotator_keys):
            raise ValueError(f"Annotator index {annotator_idx} out of range (0-{len(annotator_keys)-1})")
        ground_truth_key = annotator_keys[annotator_idx]
        print(f"Using {ground_truth_key} as ground truth")
        self.ground_truth = self.human_annotators[ground_truth_key].copy()
        # Apply filters to determine eligible question IDs
        self._apply_filters()
        self.print_filtering_stats()
        return ground_truth_key
    
    def _apply_filters(self) -> None:
        """Apply filters to determine eligible question IDs."""
        # filtered_ids = set()
        
        # for question_id, data in self.ground_truth.items():
        #     # Check unique_rating_filter (rating_multians)
        #     rating_multians = data.get("rating_multians")
        #     if rating_multians is not None:
        #         if not (self.unique_rating_filter[0] <= rating_multians <= self.unique_rating_filter[1]):
        #             continue
            
        #     # Check specific_filter (rating_osq)
        #     rating_osq = data.get("rating_osq")
        #     if rating_osq is not None:
        #         if not (self.specific_filter[0] <= rating_osq <= self.specific_filter[1]):
        #             continue
            
        #     # If passed all filters
        #     filtered_ids.add(question_id)
        
        # self.filtered_question_ids = filtered_ids

        all_question_ids = set()
        for annotator_id, annotations in self.human_annotators.items():
            all_question_ids.update(annotations.keys())
        
        print(f"Total unique questions across all annotators: {len(all_question_ids)}")
        
        # Count questions that pass filters for each annotator
        filtered_ids = {}
        
        for annotator_id, annotations in self.human_annotators.items():
            filtered_ids[annotator_id] = set()
            
            for question_id, data in annotations.items():
                # Check unique_rating_filter (rating_multians)
                rating_multians = data.get("rating_multians")
                if rating_multians is not None:
                    if not (self.unique_rating_filter[0] <= rating_multians <= self.unique_rating_filter[1]):
                        continue
                
                # Check specific_filter (rating_osq)
                rating_osq = data.get("rating_osq")
                if rating_osq is not None:
                    if not (self.specific_filter[0] <= rating_osq <= self.specific_filter[1]):
                        continue
                
                # If passed all filters
                filtered_ids[annotator_id].add(question_id)
            
            print(f"Annotator {annotator_id}: {len(filtered_ids[annotator_id])} questions pass filters")
        
        # Get the intersection of filtered IDs for all annotators
        annotator_keys = list(filtered_ids.keys())
        if len(annotator_keys) >= 2:
            common_filtered = filtered_ids[annotator_keys[0]].intersection(filtered_ids[annotator_keys[1]])
            print(f"Questions passing filters for both annotators: {len(common_filtered)}")

        self.filtered_question_ids = common_filtered

        print(f"Applied filters: {len(self.filtered_question_ids)} questions remain out of {len(self.ground_truth)}")

    def print_filtering_stats(self) -> None:
        """Print statistics about the filtering process."""
        # Get all question IDs from the human annotators
        all_question_ids = set()
        for annotator_id, annotations in self.human_annotators.items():
            all_question_ids.update(annotations.keys())
        
        print(f"Total unique questions across all annotators: {len(all_question_ids)}")
        
        # Count questions that pass filters for each annotator
        filtered_ids = {}
        
        for annotator_id, annotations in self.human_annotators.items():
            filtered_ids[annotator_id] = set()
            
            for question_id, data in annotations.items():
                # Check unique_rating_filter (rating_multians)
                rating_multians = data.get("rating_multians")
                if rating_multians is not None:
                    if not (self.unique_rating_filter[0] <= rating_multians <= self.unique_rating_filter[1]):
                        continue
                
                # Check specific_filter (rating_osq)
                rating_osq = data.get("rating_osq")
                if rating_osq is not None:
                    if not (self.specific_filter[0] <= rating_osq <= self.specific_filter[1]):
                        continue
                
                # If passed all filters
                filtered_ids[annotator_id].add(question_id)
            
            print(f"Annotator {annotator_id}: {len(filtered_ids[annotator_id])} questions pass filters")
        
        # Get the intersection of filtered IDs for all annotators
        annotator_keys = list(filtered_ids.keys())
        if len(annotator_keys) >= 2:
            common_filtered = filtered_ids[annotator_keys[0]].intersection(filtered_ids[annotator_keys[1]])
            print(f"Questions passing filters for both annotators: {len(common_filtered)}")
        
    
    def load_mcq_responses(self) -> None:
        """Load MCQ responses from the specified file."""
        mcq_data = load_jsonl_file(self.mcq_file)
        if self.is_gpqa_diamond:
            for data in mcq_data:
                question_id = str(data.get("question_id"))
                model = data.get("model")
                exact_match = data.get("exact_match")
                if question_id is not None and model is not None and exact_match is not None:
                    key = (question_id, model)
                    if isinstance(exact_match, list):
                        exact_match = exact_match[0]
                    if isinstance(exact_match, str) and exact_match in ["0", "1"]:
                        self.mcq_responses[key] = int(exact_match)
                    elif isinstance(exact_match, int) and exact_match in [0, 1]:
                        self.mcq_responses[key] = exact_match
            print(f"Loaded {len(self.mcq_responses)} MCQ (qid, model) responses")
            return
        for data in mcq_data:
            question_id = str(data.get("question_id"))
            exact_match = data.get("exact_match")
            if question_id is not None and exact_match is not None:
                if isinstance(exact_match, list):
                    exact_match = exact_match[0]
                if isinstance(exact_match, str) and exact_match in ["0", "1"]:
                    self.mcq_responses[question_id] = int(exact_match)
                elif isinstance(exact_match, int) and exact_match in [0, 1]:
                    self.mcq_responses[question_id] = exact_match

        print(f"Loaded {len(self.mcq_responses)} MCQ responses")
    
    def load_lm_matchings(self) -> None:
        """Load LM matchings from the specified file."""
        lm_data = load_jsonl_file(self.lm_matchings_file)
        if self.is_gpqa_diamond:
            for data in lm_data:
                question_id = str(data.get("question_id"))
                model = data.get("model")
                if question_id is not None and model is not None:
                    key = (question_id, model)
                    matchings = {}
                    for k, value in data.items():
                        if k.startswith("score_"):
                            matcher_name = k.replace("score_", "")
                            if matcher_name not in self.exclude_matchers:
                                self.matchers.add(matcher_name)
                                if value in ["0", "1"]:
                                    matchings[matcher_name] = int(value)
                                elif isinstance(value, int):
                                    matchings[matcher_name] = value
                                elif isinstance(value, list):
                                    matchings[matcher_name] = value[0] if len(value) > 0 else None
                                else:
                                    matchings[matcher_name] = None
                    self.lm_matchings[key] = matchings
            print(f"Loaded {len(self.lm_matchings)} LM matching (qid, model) sets")
            print(f"Found {len(self.matchers)} matchers: {', '.join(self.matchers)}")
            if self.exclude_matchers:
                print(f"Excluded matchers: {', '.join(self.exclude_matchers)}")
            return
        for data in lm_data:
            question_id = str(data.get("question_id"))
            if question_id is not None:
                matchings = {}
                for key, value in data.items():
                    if key.startswith("score_"):
                        matcher_name = key.replace("score_", "")
                        if matcher_name not in self.exclude_matchers:
                            self.matchers.add(matcher_name)
                            if value in ["0", "1"]:
                                matchings[matcher_name] = int(value)
                            elif isinstance(value, int):
                                matchings[matcher_name] = value
                            elif isinstance(value, list):
                                matchings[matcher_name] = value[0] if len(value) > 0 else None
                            else:
                                matchings[matcher_name] = None
                self.lm_matchings[question_id] = matchings
        print(f"Loaded {len(self.lm_matchings)} LM matching sets")
        print(f"Found {len(self.matchers)} matchers: {', '.join(self.matchers)}")
        if self.exclude_matchers:
            print(f"Excluded matchers: {', '.join(self.exclude_matchers)}")
    
    def load_cloze_responses(self) -> None:
        """Load MC Cloze responses from the specified file."""
        if not self.cloze_file:
            return
        cloze_data = load_jsonl_file(self.cloze_file)
        for data in cloze_data:
            question_id = str(data.get("question_id"))
            exact_match = data.get("exact_match")
            if question_id is not None and exact_match is not None:
                if isinstance(exact_match, list):
                    exact_match = exact_match[0]
                if isinstance(exact_match, str) and exact_match in ["0", "1"]:
                    self.cloze_responses[question_id] = int(exact_match)
                elif isinstance(exact_match, int) and exact_match in [0, 1]:
                    self.cloze_responses[question_id] = exact_match
        print(f"Loaded {len(self.cloze_responses)} MC Cloze responses")

    def load_verify_responses(self) -> None:
        """Load MC Verify responses from the specified file."""
        if not self.verification_file:
            return
        verify_data = load_jsonl_file(self.verification_file)
        if self.is_gpqa_diamond:
            for data in verify_data:
                question_id = str(data.get("question_id"))
                model = data.get("model")
                exact_match = data.get("exact_match")
                if question_id is not None and model is not None and exact_match is not None:
                    key = (question_id, model)
                    if isinstance(exact_match, list):
                        exact_match = exact_match[0]
                    if isinstance(exact_match, str) and exact_match in ["0", "1"]:
                        self.verify_responses[key] = int(exact_match)
                    elif isinstance(exact_match, int) and exact_match in [0, 1]:
                        self.verify_responses[key] = exact_match
            print(f"Loaded {len(self.verify_responses)} MC Verify (qid, model) responses")
            return
        for data in verify_data:
            question_id = str(data.get("question_id"))
            exact_match = data.get("exact_match")
            if question_id is not None and exact_match is not None:
                if isinstance(exact_match, list):
                    exact_match = exact_match[0]
                if isinstance(exact_match, str) and exact_match in ["0", "1"]:
                    self.verify_responses[question_id] = int(exact_match)
                elif isinstance(exact_match, int) and exact_match in [0, 1]:
                    self.verify_responses[question_id] = exact_match
        print(f"Loaded {len(self.verify_responses)} MC Verify responses")
    
    def calculate_alignment(self, ground_truth_key: str, n_bootstrap: int = 1000, normalize: str = "none") -> pd.DataFrame:
        """
        Calculate alignment between ground truth, MCQ, MC Cloze, MC Verify, and LM matchers.
        
        Args:
            ground_truth_key: Key identifying the ground truth annotator
            n_bootstrap: Number of bootstrap samples for error calculation
            normalize: Normalization method ("none", "balance", "reweight", or "scotts")
        
        Returns:
            DataFrame with alignment percentages and standard errors
        """
        # Prepare data structures
        results = defaultdict(lambda: {"agreements": [], "total": 0})
        
        # First pass to categorize questions by ground truth
        question_ids_by_gt = {0: [], 1: []}
        
        for question_id, gt_data in self.ground_truth.items():
            # Skip if not in filtered question IDs
            if question_id not in self.filtered_question_ids:
                continue
                
            # Get ground truth binary score
            gt_rating = gt_data.get("rating_match")
            gt_score = convert_rating_to_binary(gt_rating)
            
            if gt_score is None:
                print(f"Question {question_id} not in ground truth")
                continue  # Skip if ground truth is unsure (rating = 3)
            
            # Categorize by ground truth score
            question_ids_by_gt[gt_score].append(question_id)
        
        # Determine majority class for constant baseline
        majority_class = 1 if len(question_ids_by_gt[1]) >= len(question_ids_by_gt[0]) else 0
        
        # Select questions and calculate weights based on normalization method
        questions_to_use, weights = select_questions_and_calculate_weights(question_ids_by_gt, normalize)
        
        # Initialize data for Scott's Pi if needed
        scotts_data = defaultdict(lambda: {"agreements": 0, "total": 0, "gt_dist": {}, "pred_dist": {}}) if normalize == "scotts" else None
        
        # Track ground truth distribution for constant baseline
        gt_counts = {0: 0, 1: 0}
        
        # Create constant baseline predictions (always predicting the majority class)
        constant_baseline_preds = {}
        
        # For each question with ground truth
        for question_id in questions_to_use:
            gt_data = self.ground_truth[question_id]
            
            # Get ground truth binary score
            gt_rating = gt_data.get("rating_match")
            if not self.is_math:
                gt_score = convert_rating_to_binary(gt_rating)
            else:
                gt_score = gt_rating
            # Add to constant baseline predictions
            constant_baseline_preds[question_id] = majority_class
            
            # Count ground truth distribution
            gt_counts[gt_score] += 1
            
            # Default weight is 1.0 if not specified
            weight = weights.get(question_id, 1.0) if normalize == "reweight" else 1.0
            
            # Check alignment with other human annotators
            if not self.is_math:
                for annotator_id, annotations in self.human_annotators.items():
                    if annotator_id == ground_truth_key:
                        continue  # Skip ground truth annotator
                    
                    if question_id in annotations:
                        human_rating = annotations[question_id].get("rating_match")
                        human_score = convert_rating_to_binary(human_rating)
                        
                        if human_score is not None:
                            results[annotator_id]["total"] += 1
                            agreement = int(gt_score == human_score)
                            results[annotator_id]["agreements"].append(agreement * weight)
                            
                            # For Scott's Pi, collect data
                            if normalize == "scotts":
                                collect_scotts_pi_data(question_id, gt_score, human_score, scotts_data, annotator_id)
            
            # Check alignment with MCQ
            if question_id in self.mcq_responses:
                mcq_score = self.mcq_responses[question_id]
                if mcq_score is not None:
                    results["mcq"]["total"] += 1
                    agreement = int(gt_score == mcq_score)
                    results["mcq"]["agreements"].append(agreement * weight)
                    
                    # For Scott's Pi, collect data
                    if normalize == "scotts":
                        collect_scotts_pi_data(question_id, gt_score, mcq_score, scotts_data, "mcq")
            
            # Check alignment with MC Cloze
            if question_id in self.cloze_responses:
                cloze_score = self.cloze_responses[question_id]
                if cloze_score is not None:
                    results["mc_cloze"]["total"] += 1
                    agreement = int(gt_score == cloze_score)
                    results["mc_cloze"]["agreements"].append(agreement * weight)
                    if normalize == "scotts":
                        collect_scotts_pi_data(question_id, gt_score, cloze_score, scotts_data, "mc_cloze")
            
            # Check alignment with MC Verify
            if question_id in self.verify_responses:
                verify_score = self.verify_responses[question_id]
                if verify_score is not None:
                    results["mc_verify"]["total"] += 1
                    agreement = int(gt_score == verify_score)
                    results["mc_verify"]["agreements"].append(agreement * weight)
                    if normalize == "scotts":
                        collect_scotts_pi_data(question_id, gt_score, verify_score, scotts_data, "mc_verify")
            
            # Check alignment with LM matchers
            if question_id in self.lm_matchings:
                for matcher, matching in self.lm_matchings[question_id].items():
                    if matching is not None:
                        results[matcher]["total"] += 1
                        agreement = int(gt_score == matching)
                        results[matcher]["agreements"].append(agreement * weight)
                        
                        # For Scott's Pi, collect data
                        if normalize == "scotts":
                            collect_scotts_pi_data(question_id, gt_score, matching, scotts_data, matcher)
            
            # Check alignment with constant baseline
            baseline_pred = constant_baseline_preds.get(question_id)
            if baseline_pred is not None:
                results["constant_baseline"]["total"] += 1
                agreement = int(gt_score == baseline_pred)
                results["constant_baseline"]["agreements"].append(agreement * weight)
                
                # For Scott's Pi, collect data
                if normalize == "scotts":
                    collect_scotts_pi_data(question_id, gt_score, baseline_pred, scotts_data, "constant_baseline")
        
        # Calculate standard constant baseline (just for reporting)
        total_gt = sum(gt_counts.values())
        standard_constant_baseline = max(gt_counts.values()) / total_gt * 100 if total_gt > 0 else 50
        
        # Print ground truth distribution
        print(f"Ground truth distribution: {gt_counts[1]} positive, {gt_counts[0]} negative")
        print(f"Standard constant baseline (raw frequency): {standard_constant_baseline:.1f}%")
        
        # Calculate agreement percentages and bootstrap standard errors
        alignment_data = []
        normalized_constant_baseline = None
        
        for source, data in results.items():
            if data["total"] == 0:
                continue
                
            # Calculate agreement metric based on normalization method
            agreement_pct, std_error = calculate_agreement_metric(
                data,
                normalize,
                scotts_data[source] if normalize == "scotts" else None,
                n_bootstrap
            )
            
            # Store normalized constant baseline value
            if source == "constant_baseline":
                normalized_constant_baseline = agreement_pct
            
            # Determine source type
            if source.startswith("human_"):
                source_type = "Human"
            elif source == "mcq":
                source_type = "MCQ"
            elif source == "mc_cloze":
                source_type = "MC Cloze"
            elif source == "mc_verify":
                source_type = "MC Verify"
            elif source == "constant_baseline":
                source_type = "Constant Baseline"
            else:
                source_type = "Matcher"
            
            alignment_data.append({
                "Source": source,
                "Agreement (%)": agreement_pct,
                "Std Error": std_error,
                "Type": source_type,
                "Count": data["total"]
            })
        
        # Prepare predictions for error analysis
        ground_truth_scores = {}
        for question_id in questions_to_use:
            gt_data = self.ground_truth[question_id]
            gt_rating = gt_data.get("rating_match")
            if not self.is_math:
                gt_score = convert_rating_to_binary(gt_rating)
            else:
                gt_score = gt_rating
            if gt_score is not None:
                ground_truth_scores[question_id] = gt_score
        
        # Prepare predictions dictionary
        predictions = {
            "mcq": {qid: self.mcq_responses[qid] for qid in self.mcq_responses 
                   if qid in questions_to_use and self.mcq_responses[qid] is not None},
            "mc_cloze": {qid: self.cloze_responses[qid] for qid in self.cloze_responses 
                   if qid in questions_to_use and self.cloze_responses[qid] is not None},
            "mc_verify": {qid: self.verify_responses[qid] for qid in self.verify_responses 
                   if qid in questions_to_use and self.verify_responses[qid] is not None},
            "constant_baseline": constant_baseline_preds
        }
        
        # Add matcher predictions
        for matcher in self.matchers:
            matcher_preds = {}
            for qid in questions_to_use:
                if qid in self.lm_matchings and matcher in self.lm_matchings[qid]:
                    score = self.lm_matchings[qid][matcher]
                    if score is not None:
                        matcher_preds[qid] = score
            predictions[matcher] = matcher_preds
        
        # For each human annotator (excluding the ground truth one)
        if not self.is_math:
            for annotator_id, annotations in self.human_annotators.items():
                if annotator_id == ground_truth_key:
                    continue
                
                human_preds = {}
                for qid in questions_to_use:
                    if qid in annotations:
                        human_rating = annotations[qid].get("rating_match")
                        human_score = convert_rating_to_binary(human_rating)
                        if human_score is not None:
                            human_preds[qid] = human_score
                
                predictions[annotator_id] = human_preds
        
        # Run error analysis
        analyze_error_types(ground_truth_scores, predictions, questions_to_use, self.ground_truth)
        
        # Use the normalized constant baseline if available, otherwise use the standard one
        final_constant_baseline = normalized_constant_baseline if normalized_constant_baseline is not None else standard_constant_baseline
        
        return pd.DataFrame(alignment_data), final_constant_baseline

    def calculate_accuracy(self, ground_truth_key: str, n_bootstrap: int = 1000) -> pd.DataFrame:
        """
        Calculate the raw scores/accuracies provided by different evaluation methods.
        
        Args:
            ground_truth_key: Key identifying the ground truth annotator (used for filtering)
            n_bootstrap: Number of bootstrap samples for error calculation
        
        Returns:
            DataFrame with average scores and standard errors
        """
        # Prepare data structures
        results = defaultdict(lambda: {"scores": [], "total": 0})
        
        # Collect all question IDs to consider
        all_question_ids = set()
        if self.is_math:
            # For math dataset, use all questions from ground truth
            all_question_ids = set(self.ground_truth.keys())
        else:
            # For other datasets, use filtered question IDs
            all_question_ids = self.filtered_question_ids
        
        # For each question
        for question_id in all_question_ids:
            # Collect scores from other human annotators (excluding ground truth)
            if not self.is_math:
                for annotator_id, annotations in self.human_annotators.items():
                    # if annotator_id == ground_truth_key:
                    #     continue  # Skip ground truth annotator
                    
                    if question_id in annotations:
                        human_rating = annotations[question_id].get("rating_match")
                        human_score = convert_rating_to_binary(human_rating)
                        
                        if human_score is not None:
                            results[annotator_id]["total"] += 1
                            results[annotator_id]["scores"].append(human_score)
            
            # Collect scores from MCQ
            if question_id in self.mcq_responses:
                mcq_score = self.mcq_responses[question_id]
                if mcq_score is not None:
                    results["mcq"]["total"] += 1
                    results["mcq"]["scores"].append(mcq_score)
            
            # Collect scores from MC Cloze
            if question_id in self.cloze_responses:
                cloze_score = self.cloze_responses[question_id]
                if cloze_score is not None:
                    results["mc_cloze"]["total"] += 1
                    results["mc_cloze"]["scores"].append(cloze_score)
            
            # Collect scores from MC Verify
            if question_id in self.verify_responses:
                verify_score = self.verify_responses[question_id]
                if verify_score is not None:
                    results["mc_verify"]["total"] += 1
                    results["mc_verify"]["scores"].append(verify_score)
            
            # Collect scores from LM matchers
            if question_id in self.lm_matchings:
                for matcher, matching in self.lm_matchings[question_id].items():
                    if matching is not None:
                        results[matcher]["total"] += 1
                        results[matcher]["scores"].append(matching)
        
        # Calculate average scores and bootstrap standard errors
        accuracy_data = []
        
        for source, data in results.items():
            if data["total"] == 0:
                continue
                
            # Calculate average score percentage
            scores_array = np.array(data["scores"])
            avg_score_pct = np.mean(scores_array) * 100
            
            # Bootstrap to calculate standard error
            bootstrap_samples = []
            for _ in range(n_bootstrap):
                bootstrap_indices = np.random.choice(len(scores_array), size=len(scores_array), replace=True)
                bootstrap_sample = scores_array[bootstrap_indices]
                bootstrap_samples.append(np.mean(bootstrap_sample) * 100)
            
            std_error = np.std(bootstrap_samples)
            
            # Determine source type
            if source.startswith("human_"):
                source_type = "Human"
            elif source == "mcq":
                source_type = "MCQ"
            elif source == "mc_cloze":
                source_type = "MC Cloze"
            elif source == "mc_verify":
                source_type = "MC Verify"
            else:
                source_type = "Matcher"
            
            accuracy_data.append({
                "Source": source,
                "Accuracy (%)": avg_score_pct,
                "Std Error": std_error,
                "Type": source_type,
                "Count": data["total"]
            })
        
        return pd.DataFrame(accuracy_data)


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze alignment between human annotations and model matchings.')
    
    # Add arguments
    parser.add_argument('--dataset-name', type=str, default="mmlu_pro",
                        help='Dataset name for organizing plots (e.g., "mmlu_pro")')
    
    parser.add_argument('--human-annotations-dir', type=str, 
                        default=None,
                        help='Directory containing human annotation files')
    
    parser.add_argument('--mcq-file', type=str,
                        default=None,
                        help='File containing model MCQ responses')
    
    parser.add_argument('--lm-matchings-file', type=str,
                        default=None,
                        help='File containing model free-form responses and LM matchings')
    
    parser.add_argument('--cloze-file', type=str,
                        default=None,
                        help='File containing MC cloze responses')
    
    parser.add_argument('--verification-file', type=str,
                        default=None,
                        help='File containing MC verification responses')
    
    parser.add_argument('--ground-truth-idx', type=int, default=1,
                        help='Index of human annotator to use as ground truth (0-based)')
    
    parser.add_argument('--unique-rating-filter', type=int, nargs=2, default=[1, 5],
                        help='Range for filtering by rating_multians (min, max)')
    
    parser.add_argument('--specific-filter', type=int, nargs=2, default=[1, 5],
                        help='Range for filtering by rating_osq (min, max)')
    
    parser.add_argument('--no-constant-baseline', action='store_true',
                        help='Do not show the constant baseline line on the plot')
    
    parser.add_argument('--exclude-matchers', type=str, nargs='+', 
                        default=["qwen-2.5-14b-instruct", "llama-3.2-1b-instruct", "model", "llama-3.2-3b-instruct",
                                 "qwen3-8b", "Qwen3_0_6B", "llama-3-8b-instruct", "llama-3.1-8b-instruct", "llama-2-70b",
                                  "deepseek-r1-JUDGE", "Qwen3_8B", "o4-mini", "qwen-2.5-7b-instruct-JUDGE", "qwen3-14b", "qwen-2.5-72b-instruct", "llama-4-maverick"],
                        help='List of matcher models to exclude from analysis')
    
    parser.add_argument('--normalize', type=str, choices=["none", "balance", "reweight", "scotts"], default="none",
                        help='Normalization method to use ("none", "balance", "reweight", or "scotts")')
    
    parser.add_argument('--mainfig', action='store_true',
                        help='If set, produce the main vertical figure with selected models only (scotts only)')
    
    parser.add_argument('--plot-accuracy', action='store_true',
                        help='If set, generate accuracy plot instead of alignment plot')
    
    # Parse arguments
    args = parser.parse_args()

    if args.normalize == "scotts":
        args.no_constant_baseline = True
    
    if args.human_annotations_dir is None:
        args.human_annotations_dir = f"/is/cluster/fast/nchandak/qaevals/judge_outputs/alignment_plot/annotations/{args.dataset_name}"
    if args.mcq_file is None:
        args.mcq_file = f"/is/cluster/fast/nchandak/qaevals/judge_outputs/alignment_plot/mcq/{args.dataset_name}/samples.jsonl"
    if args.lm_matchings_file is None:
        args.lm_matchings_file = f"/is/cluster/fast/nchandak/qaevals/judge_outputs/alignment_plot/gen/{args.dataset_name}/samples.jsonl"
    if args.cloze_file is None:
        args.cloze_file = f"/is/cluster/fast/nchandak/qaevals/judge_outputs/alignment_plot/cloze/{args.dataset_name}/samples.jsonl"
    if args.verification_file is None:
        args.verification_file = f"/is/cluster/fast/nchandak/qaevals/judge_outputs/alignment_plot/verify/{args.dataset_name}/samples.jsonl"
    
    # Create plots directory structure: plots/human/{dataset-name}/{normalize}/
    plots_dir = os.path.join("plots", "human", args.dataset_name, args.normalize)
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir, exist_ok=True)
        print(f"Created plots directory: {plots_dir}")
    
    # Initialize analyzer with provided arguments
    analyzer = AlignmentAnalyzer(
        human_annotations_dir=args.human_annotations_dir,
        mcq_file=args.mcq_file,
        lm_matchings_file=args.lm_matchings_file,
        unique_rating_filter=tuple(args.unique_rating_filter),
        specific_filter=tuple(args.specific_filter),
        exclude_matchers=args.exclude_matchers,
        dataset_name=args.dataset_name,
        cloze_file=args.cloze_file,
        verification_file=args.verification_file
    )
    
    # Define ground truth
    ground_truth_key = analyzer.define_ground_truth(annotator_idx=args.ground_truth_idx)
    
    # Calculate alignment
    alignment_result = analyzer.calculate_alignment(ground_truth_key, normalize=args.normalize)
    alignment_df = alignment_result[0]
    constant_baseline = alignment_result[1]
    
    if args.mainfig:
        if args.normalize != "scotts":
            print("--mainfig only supported with --normalize scotts")
            return
        from utils import plot_mainfig
        plot_mainfig(
            alignment_df,
            ground_truth_key,
            constant_baseline=constant_baseline,
            output_file=f"plots/mainfig_{args.dataset_name}2.pdf",
            normalize=args.normalize,
            dataset_name=args.dataset_name
        )
        return

    accuracy_df = analyzer.calculate_accuracy(ground_truth_key)
        
    # Generate automatic output filename
    multians_filter = f"multians_{args.unique_rating_filter[0]}-{args.unique_rating_filter[1]}"
    osq_filter = f"osq_{args.specific_filter[0]}-{args.specific_filter[1]}"
    baseline = "no_baseline" if args.no_constant_baseline else "with_baseline"
    # filename = f"accuracy_alignment_{ground_truth_key}_{multians_filter}_{osq_filter}_{baseline}.pdf"
    filename = f"alignment_{ground_truth_key}_{multians_filter}_{osq_filter}_{baseline}.pdf"
    output_file = os.path.join(plots_dir, filename)
    
    # plot_accuracy_and_alignment(
    #     accuracy_df=accuracy_df,
    #     alignment_df=alignment_df,
    #     show_constant_baseline=not args.no_constant_baseline,
    #     constant_baseline=constant_baseline,
    #     ground_truth_key=ground_truth_key,
    #     output_file=output_file,
    #     normalize=args.normalize,
    #     dataset_name=args.dataset_name,
    # )

    # Plot results
    plot_alignment(
        alignment_df, 
        ground_truth_key, 
        show_constant_baseline=not args.no_constant_baseline,
        constant_baseline=constant_baseline,
        output_file=output_file,
        normalize=args.normalize,
        dataset_name=args.dataset_name
    )
    
    # if args.plot_accuracy:
    #     # Calculate and plot accuracy
    #     accuracy_df = analyzer.calculate_accuracy(ground_truth_key)
        
        
    #     # Generate automatic output filename for accuracy plot
    #     multians_filter = f"multians_{args.unique_rating_filter[0]}-{args.unique_rating_filter[1]}"
    #     osq_filter = f"osq_{args.specific_filter[0]}-{args.specific_filter[1]}"
    #     filename = f"accuracy_{ground_truth_key}_{multians_filter}_{osq_filter}.png"
    #     output_file = os.path.join(plots_dir, filename)
        
    #     # Plot accuracy results
    #     plot_accuracy(
    #         accuracy_df,
    #         ground_truth_key,
    #         output_file=output_file,
    #         dataset_name=args.dataset_name
    #     )
    #     return
    

if __name__ == "__main__":
    main()