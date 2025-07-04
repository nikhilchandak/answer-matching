#!/bin/bash

# This script runs multiple judge models on the same input file

# Ensure we have the OpenRouter API key
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "Error: OPENROUTER_API_KEY environment variable is not set."
    echo "Please set it with: export OPENROUTER_API_KEY=your_api_key"
    exit 1
fi

# Parse command line arguments
INPUT_FILE=""
OUTPUT_DIR="/fast/nchandak/qaevals/lm-similarity/judge_comparison"
BATCH_SIZE=5

# Get script directory for calling run_judge.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_JUDGE_SCRIPT="$SCRIPT_DIR/run_judge.sh"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)
            INPUT_FILE="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 -i input_file.json [-o output_dir] [-b batch_size]"
            echo ""
            echo "Options:"
            echo "  -i, --input        Input JSON file with model responses and ground truth (required)"
            echo "  -o, --output-dir   Output directory (default: /fast/nchandak/qaevals/lm-similarity/judge_comparison)"
            echo "  -b, --batch-size   Batch size for API calls (default: 5)"
            echo "  -h, --help         Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run '$0 --help' for usage information."
            exit 1
            ;;
    esac
done

# Check if input file is provided
if [ -z "$INPUT_FILE" ]; then
    echo "Error: Input file is required."
    echo "Run '$0 --help' for usage information."
    exit 1
fi

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' does not exist."
    exit 1
fi

# Make sure input_file is an absolute path
if [[ ! "$INPUT_FILE" = /* ]]; then
    INPUT_FILE="$(pwd)/$INPUT_FILE"
fi

# Create output file name based on input file
FILENAME=$(basename "$INPUT_FILE")
OUTPUT_FILENAME="${FILENAME%.*}_judgments.json"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Define models to run
MODELS=(
    "google/gemini-2.0-flash-001"
    "google/gemini-2.5-flash-preview:thinking"
    "anthropic/claude-3-5-sonnet"
    "openai/o4-mini"
)

# Run each model in sequence
echo "Running multiple judge models on: $INPUT_FILE"
echo "Results will be saved to: $OUTPUT_DIR/$OUTPUT_FILENAME"
echo "Using models: ${MODELS[*]}"

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "==================================================="
    echo "Running judge model: $MODEL"
    echo "==================================================="
    bash "$RUN_JUDGE_SCRIPT" -i "$INPUT_FILE" -o "$OUTPUT_DIR" -f "$OUTPUT_FILENAME" -b "$BATCH_SIZE" -m "$MODEL"
    
    # Add a delay between models to avoid rate limits
    sleep 5
done

echo ""
echo "All judge models have been run successfully."
echo "Final results saved to: $OUTPUT_DIR/$OUTPUT_FILENAME"
echo "This file contains scores from all models for easy comparison." 