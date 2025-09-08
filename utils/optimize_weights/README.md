# Parameter Optimization for Preprint Matching

A grid search utility to find optimal parameters for preprint-to-publication matching by evaluating different combinations of matching thresholds and weights. Supports both traditional heuristic scoring and reranking with ColBERT.

## Installation

The optimizer works with two companion scripts:
1. `preprint_match_data_files.py` - performs the preprint-to-publication matching
2. `calculate_precision_recall_f-scores.py` - calculates precision/recall metrics

Ensure both scripts have their dependencies installed in the same environment, then:

```
pip install pandas
```

## Parameter Grids

### Default Heuristic-Only Grid
The script tests combinations of the following parameters (edit `DEFAULT_PARAM_GRID` in the code to customize):
```python
DEFAULT_PARAM_GRID = {
    'min_score': [0.80, 0.85, 0.90],
    'max_score_diff': [0.03, 0.04, 0.05],
    'weight_year': [0.3, 0.4, 0.5],
    'weight_title': [2.0, 2.2, 2.4],
    'weight_author': [0.8, 1.0, 1.2]
}
```

### Reranker Weight Optimization Grid
When using `--enable-reranker` with `--optimize-reranker-weights`, the script uses a specialized grid:
```python
RERANKER_WEIGHT_GRID = {
    'min_score': [0.80, 0.85, 0.90],
    'max_score_diff': [0.03, 0.04, 0.05],
    'weight_year': [0.4],  # Fixed for reranker optimization
    'weight_title': [2.0],  # Fixed for reranker optimization
    'weight_author': [1.0],  # Fixed for reranker optimization
    'heuristic_weight': [0.2, 0.3, 0.4],
    'reranker_weight': [0.8, 0.7, 0.6],
    'reranker_batch_size': [16]
}
```

## Usage

### Basic Heuristic Optimization
```bash
python optimize_weights.py \
  -i sample_input.jsonl \
  -r ground_truth.csv \
  -o results.csv \
  -m your.email@example.com \
  -u "Your Project Name/1.0"
```

### With ColBERT Reranker (Fixed Weights)
```bash
python optimize_weights.py \
  -i sample_input.jsonl \
  -r ground_truth.csv \
  -o results.csv \
  -m your.email@example.com \
  -u "Your Project Name/1.0" \
  --enable-reranker \
  --reranker-model-path "lightonai/GTE-ModernColBERT-v1"
```

### Optimize Heuristic vs Reranker Balance
```bash
python optimize_weights.py \
  -i sample_input.jsonl \
  -r ground_truth.csv \
  -o results.csv \
  -m your.email@example.com \
  -u "Your Project Name/1.0" \
  --enable-reranker \
  --optimize-reranker-weights
```

## Arguments
Required:
- `-i, --input-sample`: Input JSONL file with preprint data
- `-r, --reference-csv`: Ground truth CSV for evaluation  
- `-o, --output-results-csv`: Output file for results
- `-m, --mailto`: Email address for Crossref API
- `-u, --user-agent`: User-Agent for API requests

Optional:
- `--matcher-script-path`: Path to matcher script (default: `preprint_match_data_files.py`)
- `--evaluator-script-path`: Path to evaluation script (default: `calculate_precision_recall_f-scores.py`)
- `--temp-dir`: Directory for temporary files (default: `temp_optim_output`)
- `--log-level`: Logging verbosity (default: `WARNING`)
- `--timeout`: Timeout in seconds for matcher script execution (auto-calculated if not specified)

Reranking Options:
- `--enable-reranker`: Enable ColBERT reranker optimization
- `--reranker-model-path`: Path/name for ColBERT model (default: `lightonai/GTE-ModernColBERT-v1`)
- `--optimize-reranker-weights`: Optimize heuristic vs reranker weight balance (requires `--enable-reranker`)

The script automatically identifies and reports the parameter combination with the highest F score value.