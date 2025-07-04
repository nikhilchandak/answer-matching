# Cost Analysis Tools

This directory contains tools for analyzing the computational costs and token usage across different evaluation approaches in LLM-based question answering systems.

## Directory Structure

### Analysis Scripts
- `across_responses.py`: Analyzes cost distribution across different response types and evaluation methods
- `token_analysis.py`: Performs detailed token usage analysis for different models and tasks
- `plots/`: Directory containing generated visualization outputs

## Features

- Token usage tracking for:
  - Multiple-choice questions (MCQ)
  - Free-form responses
  - Matcher evaluations
  - Judge evaluations
- Cost calculation based on model-specific pricing
- Visualization of cost distributions
- Per-sample cost analysis
- Comparative analysis across different evaluation methods

## Usage

### Cost Analysis Across Responses

```bash
python across_responses.py [options]

Options:
--input_dir: Directory containing response files to analyze (default: /path/to/judge_outputs)
```

The script generates:
- Token usage statistics for different question types
- Cost breakdown by model and evaluation method
- Visualization plots showing cost distribution

### Token Usage Analysis

```bash
python token_analysis.py [options]

Options:
--input_dir: Directory containing files to analyze
```

## Visualization

The tools generate various plots including:
- Cost breakdown by evaluation method
- Token usage distribution
- Comparative cost analysis across models
- Per-sample cost metrics

Plots are saved in the `plots/` directory in both PNG and PDF formats.

## Requirements

- Python 3.7+
- Required packages:
  - matplotlib
  - numpy
  - scienceplots
  - logging
  - json

## Output Format

The analysis provides detailed statistics including:
- Mean cost per sample
- Token usage by question type
- Input/output token distribution
- Model-specific cost breakdown

Example output metrics:
```
- Free-form response costs
- MCQ response costs
- Matcher evaluation costs
- Judge evaluation costs
```

## Adding New Models

To add cost analysis for new models:
1. Add model pricing in the `MODEL_COST` dictionary
2. Add model name mapping in the `MODEL_NAMES` dictionary
3. Run the analysis scripts to include the new model 