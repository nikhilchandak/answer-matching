import os
import json
import logging
from typing import List, Dict
import time # Added for potential delays

logger = logging.getLogger(__name__)

class ForecastingQuestionGenerator:
    def __init__(
        self, 
        inference_engine,
    ):
        """
        Initialize the forecasting question generator.
        
        Args:
            inference_engine: Engine for text generation (must implement BaseInference)
        """
        self.inference_engine = inference_engine
    
    def format_prompt(self, article: Dict) -> str:
        """
        Format the prompt for generating forecasting questions.
        
        Args:
            article: Article dictionary containing the content
            
        Returns:
            Formatted prompt for the LLM
        """
        source_article = f"Title: {article.get('title', '')}\n\n"
        
        if 'description' in article and article['description']:
            source_article += f"Description: {article['description']}\n\n"
            
        if 'maintext' in article and article['maintext']:
            source_article += f"Content: {article['maintext']}\n\n"
            
        if 'date_publish' in article and article['date_publish']:
            source_article += f"Published Date: {article['date_publish']}\n"
        
        prompt = f"""
**Task:** Based on the provided news article, generate **5 high quality** forecasting questions which are multiple-choice format (MCQs) with 4 options each, as JSONs.
Forecasting questions are about predicting future events. Here, the predictor will have a knowledge cutoff before the article is published and no access to the article, so a forecasting question has to be posed about information explicitly stated in the article.
The correct answer should be specified as the index of the option in the options list. The JSON format should be: 
question_title: str,  background: str, options: List[str], answer: int

**Example Format**:
{{
    "question_id": "0",
    "question_title": "Who will win the nobel prize in Literature in 2016?",
    "background": "The nobel prize in literature is awarded to authors for their outstanding contributions to literature. The prize is awarded annually by the swedish academy.",
    "options": ["Thomas Pynchon", "Bob Dylan", "Haruki Murakami", "Cormac McCarthy"],
    "answer": 1
}}

Each question must follow the structured guidelines below.

### **Guidelines for Creating Multiple-Choice Forecasting Questions**

**Title Guidelines**
- **MCQ not Binary**: The question should not be a binary yes / no question, that is, do not ask questions starting with "Will". It should be in MCQ format with 4 options. 
- **Answerable based on article**: Each question must have a definitive answer based on information explicitly stated in the article. The other 3 options must surely be incorrect, again based on information in the article.
- **Not about historical knowledge**: The question should not be about recall of facts or events known before the article publish date. 
- **Direct and Precise**: Titles must be straightforward and unambiguous, avoiding vague terms. It should be in future tense, not past or perfect. 
- **Resolution Criteria**: Include resolution criteria in the question, for example resolution dates such as "by {{month_name}}, {{year}}?" or "in {{month_name}}, {{year}}?", and source of resolution such as "based on {{news source}}", "as said by {{official name}}", etc.
- **No references to article or future information**: Do not refer to the specific article, such as by saying "in the article". The forecaster does not have access to it or any information beyond the article publish date.

**MCQ Format**
- **Faithfulness to Article**: The answer should be based on information explicitly stated in the article, and not implications or your own knowledge.
- **Overspecificity**: The question should not be about the exact amount of something, which is often difficult to predict. Instead it should be about what happened, or predicting ranges. 
- **Four Options**: Provide four distinct options with exactly one being the correct prediction. The remaining three must be incorrect.
- **Option Overlap**: The options should represent disjoint outcomes. Do not include redundant options.
- **Concise**: The options should be as concise as possible while being clear and unambiguous.

**Background Guidelines**
- **Should not help answer**: The background must not directly help answer the forecasting question. Do not include any knowledge from the article or elsewhere that helps eliminate any of the options.
- **Necessary Context**: Only include information necessary to understand the question.
- **No Additional Knowledge**: Do not add any knowledge beyond the provided article.

Please generate 5 high-quality multiple-choice forecasting questions based on the provided article with the question id as "0", "1", "2", up till "4" with each question as a separate JSON object. Do not output any other text.
Article:
{source_article}

After generating 5 questions, inside <think> </think> tags, for each MCQ including the options, check which guidelines it satisfies and which it does not, and whether it is an interesting yet predictable MCQ to forecast with knowledge before the article's publication date. The question and options should be directly answerable from the article. This would still make an interesting forecasting question because the forecaster does not have access to the article. Then within the same <think> </think> tags, decide how to rank the questions from best to worst. Finally output the ranking as a python list of question_ids for example [3, 4, 2, 0, 1] inside ```python and ```. Do not output anything after that.

**Overall Format**
question 0 json
question 1 json
question 2 json
question 3 json
question 4 json

And then the following inside <think> </think> tags:
question 0 pros and cons based on guidelines and forecastability
question 1 pros and cons based on guidelines and forecastability
question 2 pros and cons based on guidelines and forecastability
question 3 pros and cons based on guidelines and forecastability
question 4 pros and cons based on guidelines and forecastability
deciding which question is best

list of question_ids from best to worst

Do not output any other text. First the 5 question jsons, wrapped inside ```json and ```, then the thinking to decide which question is best inside <think> </think> tags, finally the question_id ranking inside ```python and ```. Do not output anything after that.
"""
        return prompt
        
    def _load_existing_results(self, output_path: str) -> Dict[str, Dict]:
        """Loads existing results from the output file."""
        if not os.path.exists(output_path):
            return {}
        
        existing_data = []
        try:
            with open(output_path, 'r') as f:
                for line in f:
                    if line.strip():  # Skip empty lines
                        try:
                            item = json.loads(line)
                            existing_data.append(item)
                        except json.JSONDecodeError:
                            logger.warning(f"Could not decode JSON line in {output_path}. Skipping line.")
                
            # Use article_url as the key for quick lookup
            return {result.get("article_url", f"missing_url_{i}"): result for i, result in enumerate(existing_data)}
        except Exception as e:
            logger.error(f"Error loading existing results from {output_path}: {e}")
            return {}

    def _append_new_results(self, new_results: List[Dict], output_path: str) -> None:
        """Appends new results to the output file without reloading existing data."""
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Append new results to the file
            with open(output_path, 'a') as f:
                for result in new_results:
                    f.write(json.dumps(result) + '\n')
                    
            logger.info(f"Appended {len(new_results)} new results to {output_path}")
        except Exception as e:
            logger.error(f"Error appending new results to {output_path}: {e}")

    async def generate_questions(self, articles: List[Dict], output_path: str, batch_size: int = 5, regenerate: bool = False) -> List[Dict]:
        """
        Generate forecasting questions based on the configured method,
        loading existing results and saving incrementally.

        Args:
            articles: List of article dictionaries
            output_path: Path to save the results to (and load from)
            batch_size: Number of articles to process in parallel
            regenerate: If True, ignore existing results and start fresh.

        Returns:
            List of results containing the generated questions
        """
        if regenerate and os.path.exists(output_path):
            logger.info(f"Regenerate flag set. Removing existing results file: {output_path}")
            os.remove(output_path)
        
        # Create a new file if it doesn't exist or if regenerate is True
        if not os.path.exists(output_path):
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                pass  # Create empty file
        
        # Load existing results once at the beginning
        existing_results_map = self._load_existing_results(output_path)
        
        # Track which articles need to be reprocessed due to empty or invalid generated questions
        to_reprocess = []
        for url, result in existing_results_map.items():
            generated_text = result.get("generated_questions", "")
            # Check if generated_questions is empty, None, or contains an error message
            if (not generated_text or 
                generated_text is None or 
                (isinstance(generated_text, str) and 
                 (generated_text.strip() == "" or "ERROR:" in generated_text))):
                to_reprocess.append(url)
                logger.info(f"Marking article with URL {url} for reprocessing due to empty or invalid generated questions")
        
        # Remove articles to be reprocessed from the existing results map
        for url in to_reprocess:
            existing_results_map.pop(url)
        
        final_results = list(existing_results_map.values())  # Start with valid existing results
        processed_urls = set(existing_results_map.keys())

        # Filter articles that haven't been processed yet or need reprocessing
        pending_articles = []
        for article in articles:
            article_url = article.get("url", "")
            # Use a placeholder if URL is missing
            if not article_url:
                content_hash = hash(article.get('title', '') + article.get('maintext', ''))
                article_url = f"no_url_{content_hash}"

            if article_url not in processed_urls or article_url in to_reprocess:
                pending_articles.append(article)

        if not pending_articles:
            logger.info("No new articles to process. All results loaded from existing file.")
            return final_results

        logger.info(f"Loaded {len(final_results)} existing results. Processing {len(pending_articles)} new articles.")

        prompts = [self.format_prompt(article) for article in pending_articles]

        # Process pending articles in batches
        for i in range(0, len(pending_articles), batch_size):
            batch_articles = pending_articles[i:i+batch_size]
            batch_prompts = prompts[i:i+batch_size]

            logger.info(f"Processing batch {i//batch_size + 1}/{(len(pending_articles) + batch_size - 1)//batch_size}...")
            # Generate completions using the inference engine
            try:
                generated_texts = await self.inference_engine.generate(batch_prompts, batch_size=len(batch_prompts))
            except Exception as e:
                logger.error(f"Error during inference engine generation for batch starting at index {i}: {e}")
                # Add placeholder results for failed batch to avoid reprocessing on next run
                generated_texts = ["ERROR: Generation failed"] * len(batch_prompts)

            # Create a list for the batch results
            batch_results = []
            
            # Pair the generated texts with article info for the current batch
            for j, article in enumerate(batch_articles):
                article_url = article.get("url", "")
                if not article_url:
                    content_hash = hash(article.get('title', '') + article.get('maintext', ''))
                    article_url = f"no_url_{content_hash}"
                
                article_result = {
                    "article_title": article.get("title", ""),
                    "article_description": article.get("description", ""),
                    "article_maintext": article.get("maintext", ""),
                    "article_url": article_url,
                    "article_date_publish": article.get("date_publish", ""),
                    "article_date_modify": article.get("date_modify", ""),
                    "article_date_download": article.get("date_download", ""),
                    "generated_questions": generated_texts[j] if j < len(generated_texts) else "ERROR: Index out of bounds",
                }
                batch_results.append(article_result)
                final_results.append(article_result)
                
                # Also add to processed_urls to avoid duplicate processing if there are duplicate articles in the input
                processed_urls.add(article_url)

            # Append only the new batch results to the file
            self._append_new_results(batch_results, output_path)

        logger.info(f"Finished processing. Total results: {len(final_results)}")
        return final_results
        
    def save_results(self, results: List[Dict], output_path: str) -> None:
        """
        Save the final generated questions to a JSONL file. (Mainly for consistency)
        This is now a full rewrite operation, used only if explicitly called.
        """
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w') as f:
                for result in results:
                    f.write(json.dumps(result) + '\n')
                    
            logger.info(f"Final save completed. Saved {len(results)} results to {output_path}")
        except Exception as e:
            logger.error(f"Error during final save to {output_path}: {e}") 