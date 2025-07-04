import pandas as pd

# Load the original dataset from HuggingFace cache
orig_path = "../../../../.cache/huggingface/hub/datasets--Idavidrein--gpqa/snapshots/90b8e5be2b1d3d2dbfe016cdab47981150600c4a/gpqa_diamond.csv"
local_path = "GPQA_Questions.csv"
output_path = "gpqa-diamond-freeform.csv"

# Read the datasets
orig = pd.read_csv(orig_path)
local = pd.read_csv(local_path)

# Merge on 'Record ID' and add 'Correct Answer' as 'Answer'
merged = pd.merge(local, orig[['Record ID', 'Correct Answer']], on='Record ID', how='left')
merged = merged.rename(columns={'Correct Answer': 'Answer'})

# Reorder columns to make 'Answer' the second column
cols = merged.columns.tolist()
cols.remove('Answer')
cols.insert(1, 'Answer')
merged = merged[cols]

# Save the new dataset
merged.to_csv(output_path, index=False)
