
## How to run annotation interface

Installation needed: only flask, datasets -- pip install flask, datasets

```
cd src/visualize_resps
python visualizer.py --input_path <path_to_jsonl_file> --annotator_id <annotator_id> (default: "1") --port <port_number> (default: 5000) --debug <number_of_questions (prefix of input jsonl file, skip to load all)> --multi_model (optional, use for gpqa)
```

For ameya: To pre-load Ameya GPQA labeling, annotator_id should contain the string "ameya". Simple example launch command for ease:
```
python visualizer.py --input_path=../../gpqa_combined_samples_to_annotate.jsonl --port 5042 --annotator_id=ameya_gpqa --multi_model
```

Save path: `annotation/saves/<annotator_id>.jsonl`
Annotations will keep getting updated/saved as you navigate through the questions. Navigation won't be allowed for partially filled questions. 
