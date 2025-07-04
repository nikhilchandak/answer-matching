import datasets
from tqdm import tqdm
import hashlib
from math_verify import parse, verify

def check_math_datasets():
    # Load both datasets
    print("Loading datasets...")
    math_mc = datasets.load_dataset("guipenedo/math-mc")["test"]
    math = datasets.load_dataset("DigitalLearningGmbH/MATH-lighteval")["test"]
    
    print(f"math_mc dataset size: {len(math_mc)}")
    print(f"math dataset size: {len(math)}")
    
    # Extract questions from both datasets
    math_mc_questions = set(item["Question"] for item in math_mc)
    math_questions = set(item["problem"] for item in math)
    
    # Check if all questions in math dataset are in math_mc dataset
    missing_questions = []
    for question in tqdm(math_mc_questions, desc="Checking questions"):
        if question not in math_questions:
            missing_questions.append(question)
    
    # Print results
    if missing_questions:
        print(f"Found {len(missing_questions)} questions in math_mc dataset that are not in math dataset")
        print("First 5 missing questions:")
        for q in missing_questions[:5]:
            print(f"- {q[:100]}...")
        assert False, f"{len(missing_questions)} questions from math_mc dataset are missing in math dataset"
    else:
        print("All questions in math_mc dataset are present in math dataset")
        # return True
    
    # Rename the problem/Question column to question
    math = math.rename_column("problem", "Question")
    
    # Capitalize level, solution, type 
    math = math.rename_column("level", "Level")
    math = math.rename_column("solution", "Solution")
    math = math.rename_column("type", "Type")
    
    # Create a mapping of questions to numeric IDs
    print("Creating numeric question IDs...")
    all_questions = list(math_questions)  # Use math_questions since we confirmed all math_mc questions are in it
    question_to_id = {question: idx for idx, question in enumerate(all_questions)}
    
    def add_question_id(example):
        example["Question_ID"] = question_to_id[example["Question"]]
        return example
    
    # Process datasets
    print("Adding Question IDs to math dataset...")
    math_processed = math.map(add_question_id)
    
    print("Adding Question IDs to math_mc dataset...")
    math_mc_processed = math_mc.map(add_question_id)
    
    # Add this before pushing to hub
    print("\nSample from MATH_free:")
    print(math_processed[0])
    print("\nSample from MATH_mc:")
    print(math_mc_processed[0])
    
    # # Push to Hugging Face Hub
    # print("Pushing MATH_free to Hugging Face Hub...")
    # math_processed.push_to_hub("nikhilchandak/MATH_free")
    
    # print("Pushing MATH_mc to Hugging Face Hub...")
    # math_mc_processed.push_to_hub("nikhilchandak/MATH_mc")
    
    print("Done!")


def check_verify():
    
    exp1 = "$12, 16, 18$"
    exp2 = "$12, 18, 16$"
    
    
    # math_verify
    res = verify(parse(exp1), parse(exp2), strict=False)
    mathval = 1 if res else 0
    
    print(exp1 == exp2)
    print(mathval)
    
if __name__ == "__main__":
    # check_math_datasets()
    check_verify()
