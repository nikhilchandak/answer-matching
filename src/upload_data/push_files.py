import pandas as pd
from datasets import Dataset, DatasetDict

def main():
    # Load the filtered and unfiltered CSVs
    filtered = pd.read_csv("gpqa-diamond-freeform-filtered.csv")
    unfiltered = pd.read_csv("gpqa-diamond-freeform.csv")

    # Create Hugging Face Datasets
    ds_filtered = Dataset.from_pandas(filtered)
    ds_unfiltered = Dataset.from_pandas(unfiltered)

    # Create DatasetDict with two splits
    ds_dict = DatasetDict({
        "test": ds_filtered,
        "unfiltered": ds_unfiltered,
    })

    # Push to the hub
    ds_dict.push_to_hub("nikhilchandak/GPQA-diamond-free")

if __name__ == "__main__":
    main()
