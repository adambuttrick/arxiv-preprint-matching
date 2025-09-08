import os
import csv
import sys
import json
import time
import argparse
import itertools
import subprocess
import pandas as pd
from datetime import datetime


DEFAULT_PARAM_GRID = {
    'min_score': [0.80, 0.85, 0.90],
    'max_score_diff': [0.03, 0.04, 0.05],
    'weight_year': [0.3, 0.4, 0.5],
    'weight_title': [2.0, 2.2, 2.4],
    'weight_author': [0.8, 1.0, 1.2]
}

RERANKER_WEIGHT_GRID = {
    'min_score': [0.80, 0.85, 0.90],
    'max_score_diff': [0.03, 0.04, 0.05],
    'weight_year': [0.4],
    'weight_title': [2.0],
    'weight_author': [1.0],
    'heuristic_weight': [0.2, 0.3, 0.4],
    'reranker_weight': [0.8, 0.7, 0.6],
    'reranker_batch_size': [16]
}

MATCHER_SCRIPT = "preprint_match_data_files.py"
EVALUATOR_SCRIPT = "calculate_precision_recall_f-scores.py"

BASE_FIELDS = [
    "Status", "Error", "TP", "FP", "FN", "Precision", "Recall",
    "F0.5", "F1", "F1.5", "Positive References", "Positive Predictions"
]


def run_command(command_list, timeout=1800):
    try:
        print(f"Running command: {' '.join(command_list)}")
        process = subprocess.run(
            command_list,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout
        )
        print(f"Command finished with code: {process.returncode}")
        if process.returncode != 0:
            if process.stdout:
                print("Subprocess STDOUT (truncated):")
                print(process.stdout[:500] +
                      ('...' if len(process.stdout) > 500 else ''))
            if process.stderr:
                print("Subprocess STDERR (truncated):")
                print(process.stderr[:500] +
                      ('...' if len(process.stderr) > 500 else ''))
        elif process.stderr:
            print("Subprocess STDERR (potentially warnings):")
            print(process.stderr[:500] +
                  ('...' if len(process.stderr) > 500 else ''))

        return process.stdout, process.stderr, process.returncode
    except subprocess.TimeoutExpired:
        print(f"Error: Command timed out after {timeout} seconds: {' '.join(command_list)}")
        return None, "TimeoutExpired", -1
    except Exception as e:
        print(f"Error running command {' '.join(command_list)}: {e}")
        return None, str(e), -1


def generate_parameter_combinations(grid):
    keys, values = zip(*grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return combinations


def main():
    parser = argparse.ArgumentParser(
        description="Optimize preprint matching strategy weights by iterating through parameter combinations."
    )
    parser.add_argument(
        "-i", "--input-sample", required=True,
        help="Path to the input SAMPLE JSONL file (e.g., 100 records)."
    )
    parser.add_argument(
        "-r", "--reference-csv", required=True,
        help="Path to the reference CSV file (ground truth) for evaluation."
    )
    parser.add_argument(
        "-o", "--output-results-csv", required=True,
        help="Path to save the CSV file containing parameters and evaluation metrics for each run."
    )
    parser.add_argument(
        "-m", "--mailto", required=True,
        help="Email address for Crossref API politeness (passed to matcher script)."
    )
    parser.add_argument(
        "-u", "--user-agent", required=True,
        help="User-Agent string for Crossref API requests (passed to matcher script)."
    )
    parser.add_argument(
        "--matcher-script-path", default=MATCHER_SCRIPT,
        help=f"Path to the preprint_match_data_files.py script (default: {MATCHER_SCRIPT})."
    )
    parser.add_argument(
        "--evaluator-script-path", default=EVALUATOR_SCRIPT,
        help=f"Path to the calculate_precision_recall_f-scores.py script (default: {EVALUATOR_SCRIPT})."
    )
    parser.add_argument(
        "--temp-dir", default="temp_optim_output",
        help="Directory to store temporary output files for each run (default: temp_optim_output)."
    )
    parser.add_argument(
        "--log-level", default="WARNING",
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL', 'NONE'],
        help="Set the logging level for the matcher script subprocess (default: WARNING)."
    )

    parser.add_argument(
        "--enable-reranker", action='store_true',
        help="Enable ColBERT reranker optimization (test with reranker enabled)."
    )
    parser.add_argument(
        "--reranker-model-path", default='lightonai/GTE-ModernColBERT-v1',
        help="Path or HuggingFace model name for ColBERT reranker (default: lightonai/GTE-ModernColBERT-v1)."
    )
    parser.add_argument(
        "--optimize-reranker-weights", action='store_true',
        help="Optimize heuristic vs reranker weight balance (requires --enable-reranker)."
    )
    parser.add_argument(
        "--timeout", type=int, default=None,
        help="Override timeout in seconds for matcher script (default: auto-calculated based on dataset size)."
    )

    args = parser.parse_args()

    if args.optimize_reranker_weights and not args.enable_reranker:
        print("Error: --optimize-reranker-weights requires --enable-reranker")
        sys.exit(1)

    args.input_sample = os.path.abspath(args.input_sample)
    args.reference_csv = os.path.abspath(args.reference_csv)
    args.output_results_csv = os.path.abspath(args.output_results_csv)
    args.matcher_script_path = os.path.abspath(args.matcher_script_path)
    args.evaluator_script_path = os.path.abspath(args.evaluator_script_path)
    args.temp_dir = os.path.abspath(args.temp_dir)

    os.makedirs(args.temp_dir, exist_ok=True)
    print(f"Using temporary directory: {args.temp_dir}")

    if args.enable_reranker and args.optimize_reranker_weights:
        PARAM_GRID = RERANKER_WEIGHT_GRID
        print("Optimizing with reranker enabled and weight balance optimization")
    else:
        PARAM_GRID = DEFAULT_PARAM_GRID
        if args.enable_reranker:
            print("Running with reranker enabled using default weights")
        else:
            print("Optimizing heuristic-only parameters")

    ALL_POSSIBLE_FIELDS = list(PARAM_GRID.keys()) + BASE_FIELDS

    param_combinations = generate_parameter_combinations(PARAM_GRID)
    total_runs = len(param_combinations)
    print(f"Generated {total_runs} parameter combinations to test.")
    print(f"Results will be saved iteratively to: {args.output_results_csv}")

    output_file_exists = os.path.isfile(args.output_results_csv)
    write_header = not output_file_exists or os.path.getsize(
        args.output_results_csv) == 0

    start_time_total = time.time()

    for i, params in enumerate(param_combinations):
        run_number = i + 1
        print(f"\n--- Starting Run {run_number}/{total_runs} ---")
        print(f"Parameters: {params}")
        start_time_run = time.time()

        result_row = {**params}

        temp_matcher_output_csv = os.path.abspath(os.path.join(args.temp_dir, f"run_{run_number}_matches.csv"))

        matcher_cmd = [
            sys.executable,
            args.matcher_script_path,
            "--input", args.input_sample,
            "--output", temp_matcher_output_csv,
            "--format", "csv",
            "--mailto", args.mailto,
            "--user-agent", args.user_agent,
            "--log-level", args.log_level,
            "--min-score", str(params['min_score']),
            "--max-score-diff", str(params['max_score_diff']),
            "--weight-year", str(params['weight_year']),
            "--weight-title", str(params['weight_title']),
            "--weight-author", str(params['weight_author']),
        ]

        if args.enable_reranker:
            matcher_cmd.extend([
                "--enable-reranker",
                "--reranker-model-path", args.reranker_model_path,
            ])

            if 'heuristic_weight' in params:
                matcher_cmd.extend([
                    "--heuristic-weight", str(params['heuristic_weight']),
                    "--reranker-weight", str(params['reranker_weight']),
                ])
            if 'reranker_batch_size' in params:
                matcher_cmd.extend([
                    "--reranker-batch-size", str(
                        params['reranker_batch_size']),
                ])

        if args.timeout:
            estimated_timeout = args.timeout
            print(f"Using user-specified timeout of {estimated_timeout} seconds")
        else:
            timeout_per_entry = 2 if args.enable_reranker else 0.5
            min_timeout = 300

            try:
                with open(args.input_sample, 'r') as f:
                    num_entries = sum(1 for line in f if line.strip())
                estimated_timeout = max(min_timeout, int(
                    num_entries * timeout_per_entry))
                print(f"Using auto-calculated timeout of {estimated_timeout} seconds for {num_entries} entries")
            except:
                estimated_timeout = 1800
                print(f"Could not count entries, using default timeout of {estimated_timeout} seconds")

        matcher_stdout, matcher_stderr, matcher_retcode = run_command(
            matcher_cmd, timeout=estimated_timeout)

        if matcher_retcode != 0:
            print(f"Error: Matcher script failed for run {run_number}. Skipping evaluation.")
            result_row["Status"] = "Matcher Failed"
            result_row["Error"] = matcher_stderr[:500]

        elif not os.path.exists(temp_matcher_output_csv):
            print(f"Error: Matcher script completed but output file '{temp_matcher_output_csv}' not found for run {run_number}. Skipping evaluation.")
            result_row["Status"] = "Matcher Output Missing"

        else:
            actual_output_csv = temp_matcher_output_csv
            if os.path.isdir(temp_matcher_output_csv):
                csv_files = [f for f in os.listdir(
                    temp_matcher_output_csv) if f.endswith('.csv')]
                if csv_files:
                    actual_output_csv = os.path.join(
                        temp_matcher_output_csv, csv_files[0])
                    print(f"Found output CSV in directory: {actual_output_csv}")
                else:
                    print(f"Error: No CSV file found in directory '{temp_matcher_output_csv}' for run {run_number}. Skipping evaluation.")
                    result_row["Status"] = "Matcher Output Missing"
                    actual_output_csv = None

            if actual_output_csv:
                evaluator_cmd = [
                    sys.executable,
                    args.evaluator_script_path,
                    "--reference_csv", args.reference_csv,
                    "--test_csv", actual_output_csv,
                    "--json-output"
                ]

                eval_stdout, eval_stderr, eval_retcode = run_command(
                    evaluator_cmd)

                if eval_retcode != 0 or not eval_stdout:
                    print(f"Error: Evaluator script failed or produced no output for run {run_number}.")
                    result_row["Status"] = "Evaluator Failed"
                    result_row["Error"] = eval_stderr[:500]

                else:

                    try:
                        metrics = json.loads(eval_stdout)
                        if "error" in metrics:
                            print(f"Error reported by evaluator: {metrics['error']}")
                            result_row["Status"] = "Evaluator JSON Error"
                            result_row["Error"] = metrics['error']
                        else:

                            result_row.update(metrics)
                            result_row["Status"] = "Success"

                    except json.JSONDecodeError:
                        print(f"Error: Could not decode JSON output from evaluator for run {run_number}.")
                        print(f"Evaluator STDOUT was: {eval_stdout}")
                        result_row["Status"] = "Evaluator Output Invalid"
                        result_row["Error"] = "JSONDecodeError"

        try:

            with open(args.output_results_csv, 'a', encoding='utf-8') as csvfile:

                writer = csv.DictWriter(
                    csvfile, fieldnames=ALL_POSSIBLE_FIELDS, extrasaction='ignore')

                if write_header:
                    writer.writeheader()
                    write_header = False

                formatted_row = {k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in result_row.items()}
                writer.writerow(formatted_row)
            print(f"Run {run_number} result appended to {args.output_results_csv}")

        except IOError as e:
            print(f"Error: Could not write results for run {run_number} to CSV '{args.output_results_csv}': {e}", file=sys.stderr)

        run_duration = time.time() - start_time_run
        print(f"Run {run_number} finished in {run_duration:.2f} seconds.")

    total_duration = time.time() - start_time_total
    print(f"\n--- Optimization Complete ---")
    print(f"Total time: {total_duration:.2f} seconds.")
    print(f"All run results saved in: {args.output_results_csv}")

    try:
        results_df = pd.read_csv(args.output_results_csv)
        print(f"\nRead {len(results_df)} results back from CSV for analysis.")
    except FileNotFoundError:
        print(f"Error: Output results file '{args.output_results_csv}' not found for final analysis.", file=sys.stderr)
        return
    except Exception as e:
        print(f"Error reading results file '{args.output_results_csv}' for analysis: {e}", file=sys.stderr)
        return

    successful_runs = results_df[results_df['Status'] == 'Success'].copy()

    if successful_runs.empty:
        print("\nNo successful runs found in the results file to determine the best run.")
        return

    metric_cols_for_sort = ['F1', 'Precision', 'Recall']
    for col in metric_cols_for_sort:
        if col not in successful_runs.columns:
            print(f"Warning: Metric column '{col}' needed for sorting is missing. Cannot determine best run accurately.")

            if col == 'F1':
                return
            metric_cols_for_sort.remove(col)
        else:

            successful_runs[col] = pd.to_numeric(
                successful_runs[col], errors='coerce')

    successful_runs = successful_runs.dropna(subset=['F1'])

    if successful_runs.empty:
        print("\nNo successful runs with valid F1 scores found after conversion.")
        return

    valid_sort_columns = [col for col in [
        'F1', 'Precision', 'Recall'] if col in successful_runs.columns]
    successful_runs = successful_runs.sort_values(
        by=valid_sort_columns,
        ascending=[False] * len(valid_sort_columns)
    )
    best_run = successful_runs.iloc[0]

    print("\n--- Best Run (Max F1 Score) ---")
    print("Parameters:")
    param_cols = list(PARAM_GRID.keys())
    for p in param_cols:
        value = best_run.get(p, 'N/A')
        if isinstance(value, float):
            print(f"  {p}: {value:.4f}")
        else:
            print(f"  {p}: {value}")
    print("Metrics:")

    print(f"  Precision: {best_run.get('Precision', 'N/A'):.4f}")
    print(f"  Recall:    {best_run.get('Recall', 'N/A'):.4f}")
    print(f"  F1 Score:  {best_run.get('F1', 'N/A'):.4f}")

    tp = pd.to_numeric(best_run.get('TP'), errors='coerce')
    fp = pd.to_numeric(best_run.get('FP'), errors='coerce')
    fn = pd.to_numeric(best_run.get('FN'), errors='coerce')
    print(f"  (TP={int(tp) if pd.notna(tp) else 'N/A'}, FP={int(fp) if pd.notna(fp) else 'N/A'}, FN={int(fn) if pd.notna(fn) else 'N/A'})")


if __name__ == "__main__":
    main()
