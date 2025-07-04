from huggingface_hub import HfApi

def main():
    repo_id = "nikhilchandak/answer-matching"
    print(f"Deleting dataset repo: {repo_id}")
    api = HfApi()
    api.delete_repo(repo_id=repo_id, repo_type="dataset")
    print("Repo deleted!")

if __name__ == "__main__":
    main() 