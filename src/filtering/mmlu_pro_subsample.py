import json
import os
import random
from collections import defaultdict, Counter

def analyze_mmlu_pro_ratings():
    """
    Analyze MMLU-Pro questions based on LLM judge ratings for free-form answerability.
    Prints statistics on how many questions remain at different rating thresholds.
    """
    # Path to the MMLU-Pro question hash file
    file_path = "/fast/nchandak/qaevals/filter/mmlupro/MMLU-Pro_question_hash.jsonl"
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return
    
    # Load the data
    questions = []
    with open(file_path, 'r') as f:
        for line in f:
            questions.append(json.loads(line))
    
    print(f"Loaded {len(questions)} questions from MMLU-Pro dataset")
    
    # Count questions with valid ratings
    valid_ratings = 0
    rating_distribution = defaultdict(int)
    
    for question in questions:
        question_id = list(question.keys())[0]
        if "llm_judge_fine" in question[question_id]:
            valid_ratings += 1
            rating = question[question_id]["llm_judge_fine"]
            rating_distribution[rating] += 1
    
    print(f"Found {valid_ratings} questions with LLM judge ratings")
    # Calculate and print statistics for different thresholds
    thresholds = [7, 8, 9]
    for threshold in thresholds:
        count = sum(rating_distribution[rating] for rating in rating_distribution if rating >= threshold)
        percentage = (count / valid_ratings * 100) if valid_ratings > 0 else 0
        print(f"Rating >= {threshold}: {count} questions ({percentage:.2f}% of rated questions)")
    
    # Filter questions with rating >= 8
    filtered_questions = []
    for question in questions:
        question_id = list(question.keys())[0]
        if "llm_judge_fine" in question[question_id] and question[question_id]["llm_judge_fine"] >= 8:
            filtered_questions.append(question_id)
    
    print(f"Found {len(filtered_questions)} questions with rating >= 8")
    # Save filtered question IDs to file
    filtered_path = "/fast/nchandak/qaevals/filter/mmlupro/filtered_ids.txt"
    try:
        with open(filtered_path, 'w') as f:
            for question_id in filtered_questions:
                f.write(f"{question_id}\n")
        print(f"Saved filtered question IDs to {filtered_path}")
    except PermissionError:
        print(f"Permission denied when writing to {filtered_path}. Check file permissions.")
        # Use a different path in the user's home directory
        filtered_path = os.path.expanduser("data/mmlu_pro/filtered_ids.txt")
        with open(filtered_path, 'w') as f:
            for question_id in filtered_questions:
                f.write(f"{question_id}\n")
        print(f"Saved filtered question IDs to {filtered_path} instead")
    
    # Randomly sample 1000 IDs
    import random
    random.seed(12345)  # For reproducibility
    
    if len(filtered_questions) >= 1000:
        sampled_questions = random.sample(filtered_questions, 1000)
    else:
        print(f"Warning: Only {len(filtered_questions)} questions available, using all of them")
        sampled_questions = filtered_questions
    
    # Save sampled question IDs to file
    sample_path = "/fast/nchandak/qaevals/filter/mmlupro/filtered_random_sample_1000.txt"
    try:
        with open(sample_path, 'w') as f:
            for question_id in sampled_questions:
                f.write(f"{question_id}\n")
                
    except PermissionError:
        print(f"Permission denied when writing to {sample_path}. Check file permissions.")
        # Use a different path in the user's home directory
        sample_path = os.path.expanduser("data/mmlu_pro/filtered_random_sample_1000.txt")
        with open(sample_path, 'w') as f:
            for question_id in sampled_questions:
                f.write(f"{question_id}\n")
        print(f"Saved sampled question IDs to {sample_path} instead")
        
    print(f"Saved {len(sampled_questions)} randomly sampled question IDs to {sample_path}")

def verify_sampled_ids():
    """
    Verify that all sampled IDs are present in the filtered IDs file.
    This function ensures data integrity between the filtered and sampled datasets.
    """
    # Define paths for both files
    filtered_path = "/fast/nchandak/qaevals/filter/mmlupro/filtered_ids.txt"
    sample_path = "/fast/nchandak/qaevals/filter/mmlupro/filtered_random_sample_1000.txt"
    
    # Try the default paths first, then fall back to local paths if needed
    if not os.path.exists(filtered_path):
        filtered_path = os.path.expanduser("data/mmlu_pro/filtered_ids.txt")
    
    if not os.path.exists(sample_path):
        sample_path = os.path.expanduser("data/mmlu_pro/filtered_random_sample_1000.txt")
    
    # Read filtered IDs
    try:
        with open(filtered_path, 'r') as f:
            filtered_ids = set(line.strip() for line in f if line.strip())
        print(f"Read {len(filtered_ids)} filtered IDs from {filtered_path}")
    except FileNotFoundError:
        print(f"Error: Filtered IDs file not found at {filtered_path}")
        return False
    
    # Read sampled IDs
    try:
        with open(sample_path, 'r') as f:
            sampled_ids = [line.strip() for line in f if line.strip()]
        print(f"Read {len(sampled_ids)} sampled IDs from {sample_path}")
    except FileNotFoundError:
        print(f"Error: Sampled IDs file not found at {sample_path}")
        return False
    
    # Verify all sampled IDs are in the filtered set
    missing_ids = [id for id in sampled_ids if id not in filtered_ids]
    
    if missing_ids:
        print(f"Error: Found {len(missing_ids)} IDs in sample that are not in the filtered set:")
        for id in missing_ids[:10]:  # Show first 10 missing IDs
            print(f"  - {id}")
        if len(missing_ids) > 10:
            print(f"  ... and {len(missing_ids) - 10} more")
        return False
    else:
        print(f"Success: All {len(sampled_ids)} sampled IDs are present in the filtered set")
        return True

def create_stratified_sample():
    """
    Create a stratified sample of 1000 questions from MMLU-Pro with rating >= 8.
    Sample X questions from each category (subject) to get a total of 1000 questions.
    If a category has <= X questions, include all of them.
    """
    # Path to the MMLU-Pro question hash file
    file_path = "/fast/nchandak/qaevals/filter/mmlupro/MMLU-Pro_question_hash.jsonl"
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return
    
    # Load the data
    questions = []
    with open(file_path, 'r') as f:
        for line in f:
            questions.append(json.loads(line))
    
    print(f"Loaded {len(questions)} questions from MMLU-Pro dataset")
    
    # Load actual samples
    samples_path = "/fast/nchandak/qaevals/filtered_outputs/mmlu_pro_free/Qwen3-14B_non_thinking/samples.json"
    samples = []

    try:
        with open(samples_path, 'r') as f:
            samples = json.load(f)
        print(f"Loaded {len(samples)} samples from Qwen3-14B output")
    except FileNotFoundError:
        print(f"Error: File not found at {samples_path}")
        samples = []

    # print(questions)
    # Create a mapping from question ID to category
    id_to_category = {}
    for question in samples:
        question_id = int(question.get("question_id"))
        category = question.get("category")
        id_to_category[question_id] = category
    
    filtered_questions = []
    categories = {}
    for question in questions:
        question_id = list(question.keys())[0]
        question_data = question[question_id]
        
        if "llm_judge_fine" in question_data and question_data["llm_judge_fine"] >= 8:
            question_id = int(question_id)
            filtered_questions.append(question_id)
            # Get the category/subject
            if question_id in id_to_category:
                category = id_to_category[question_id]
                if category not in categories:
                    categories[category] = []
                categories[category].append(question_id)
    
    # Print category statistics
    print(f"Found {len(categories)} categories with questions rated >= 8:")
    for category, ids in sorted(categories.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"  - {category}: {len(ids)} questions")
    
    # Find the smallest X such that sampling min(X, len(category)) from each category
    # results in approximately 1000 questions total
    total_questions = 1000
    num_categories = len(categories)
    
    # Binary search to find the optimal X
    min_x = 1
    max_x = total_questions  # Upper bound
    optimal_x = None
    
    while min_x <= max_x:
        mid_x = (min_x + max_x) // 2
        
        # Calculate how many questions we'd get with this X
        total_with_x = sum(min(mid_x, len(ids)) for ids in categories.values())
        
        if total_with_x < total_questions:
            min_x = mid_x + 1
        elif total_with_x > total_questions:
            max_x = mid_x - 1
        else:
            optimal_x = mid_x
            break
    
    # If we didn't find an exact match, use the closest value
    if optimal_x is None:
        # Recalculate with max_x (which will be < 1000) and min_x (which will be > 1000)
        total_with_max_x = sum(min(max_x, len(ids)) for ids in categories.values())
        total_with_min_x = sum(min(min_x, len(ids)) for ids in categories.values())
        
        # Choose the one that's closest to 1000
        if abs(total_with_max_x - total_questions) <= abs(total_with_min_x - total_questions):
            optimal_x = max_x
            total_sampled = total_with_max_x
        else:
            optimal_x = min_x
            total_sampled = total_with_min_x
    else:
        total_sampled = total_questions
    
    print(f"Optimal X: {optimal_x}, which will result in {total_sampled} questions")
    
    # Sample questions using the optimal X
    random.seed(12345)  # For reproducibility
    sampled_questions = []
    
    for category, ids in categories.items():
        num_to_sample = min(optimal_x, len(ids))
        if num_to_sample == len(ids):
            category_sample = ids
        else:
            category_sample = random.sample(ids, num_to_sample)
        
        sampled_questions.extend(category_sample)
        print(f"Sampled {len(category_sample)} questions from {category}")
    
    # Randomly shuffle the sampled questions
    random.shuffle(sampled_questions)
    print(f"Total sampled: {len(sampled_questions)} questions")
    
    # Save stratified sample to file
    sample_path = f"/fast/nchandak/qaevals/filter/mmlupro/filtered_stratified_sample_{len(sampled_questions)}.txt"
    try:
        with open(sample_path, 'w') as f:
            for question_id in sampled_questions:
                f.write(f"{question_id}\n")
        print(f"Saved stratified sample to {sample_path}")
    except PermissionError:
        print(f"Permission denied when writing to {sample_path}. Check file permissions.")
        # Use a different path in the user's home directory
        sample_path = os.path.expanduser(f"data/mmlu_pro/filtered_stratified_sample_{len(sampled_questions)}.txt")
        with open(sample_path, 'w') as f:
            for question_id in sampled_questions:
                f.write(f"{question_id}\n")
        print(f"Saved stratified sample to {sample_path} instead")
    
    return sampled_questions


if __name__ == "__main__":
    # analyze_mmlu_pro_ratings()
    # verify_sampled_ids()
    create_stratified_sample()
