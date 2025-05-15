import os
import csv
import sys
import gzip
import json
import logging
import argparse
from urllib.parse import urlparse
from strategies.preprint_sbmv_datacite.strategy import PreprintSbmvStrategy
from matching.utils import (
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_MAX_RETRIES,
    DEFAULT_BACKOFF_FACTOR,
    DEFAULT_STATUS_FORCELIST
)

DEFAULT_MAX_CONSECUTIVE_LINE_FAILURES = 10
DEFAULT_MAX_CONSECUTIVE_FILE_FAILURES = 3


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Recursively find and process JSONL files (.jsonl, .jsonl.gz) in an input directory, "
                    "outputting results to a corresponding structure in an output directory."
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to the input directory containing .jsonl or .jsonl.gz files."
    )
    parser.add_argument(
        "-o", "--output",
        required=False,
        default='./output',
        help="Path to the output directory where results will be saved. Will be created if it doesn't exist (default: ./output)."
    )
    parser.add_argument(
        "-f", "--format",
        required=True,
        choices=['json', 'csv'],
        help="Output format ('json' or 'csv') for the result files."
    )
    parser.add_argument(
        "-m", "--mailto",
        required=True,
        help="Email address for Crossref API politeness."
    )
    parser.add_argument(
        "-u", "--user-agent",
        required=True,
        help="User-Agent string for Crossref API requests."
    )

    parser.add_argument(
        "-ll", "--log-level",
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL', 'NONE'],
        help="Set the logging level (default: INFO). 'NONE' disables logging."
    )
    parser.add_argument(
        "-lf", "--log-file",
        help="Optional: Path to a file to write logs to. If not provided, logs go to stderr."
    )
    parser.add_argument(
        "-lc", "--log-candidates",
        action='store_true',
        help="If set, log raw Crossref candidate results to the candidate log file (appended for all files)."
    )
    parser.add_argument(
        "-cf", "--candidate-log-file",
        default="crossref_candidates.log",
        help="Path to the file for logging raw candidates (default: crossref_candidates.log)."
    )

    strategy_defaults = {
        'min_score': PreprintSbmvStrategy.DEFAULT_MIN_SCORE,
        'max_score_diff': PreprintSbmvStrategy.DEFAULT_MAX_SCORE_DIFF,
        'weight_year': PreprintSbmvStrategy.DEFAULT_WEIGHT_YEAR,
        'weight_title': PreprintSbmvStrategy.DEFAULT_WEIGHT_TITLE,
        'weight_author': PreprintSbmvStrategy.DEFAULT_WEIGHT_AUTHOR,
        'max_query_len': PreprintSbmvStrategy.DEFAULT_MAX_QUERY_LEN
    }
    parser.add_argument('--min-score', type=float, default=strategy_defaults['min_score'], help=f"Minimum score threshold for a match (default: {strategy_defaults['min_score']})")
    parser.add_argument('--max-score-diff', type=float, default=strategy_defaults['max_score_diff'], help=f"Maximum allowed score difference from top score for multiple matches (default: {strategy_defaults['max_score_diff']})")
    parser.add_argument('--weight-year', type=float, default=strategy_defaults['weight_year'], help=f"Weight for the year score component (default: {strategy_defaults['weight_year']})")
    parser.add_argument('--weight-title', type=float, default=strategy_defaults['weight_title'], help=f"Weight for the title score component (default: {strategy_defaults['weight_title']})")
    parser.add_argument('--weight-author', type=float, default=strategy_defaults['weight_author'], help=f"Weight for the author score component (default: {strategy_defaults['weight_author']})")
    parser.add_argument('--max-query-len', type=int, default=strategy_defaults['max_query_len'], help=f"Maximum length of the query string sent to Crossref (default: {strategy_defaults['max_query_len']})")

    parser.add_argument(
        '--timeout', type=float, nargs=2, metavar=('CONNECT_TIMEOUT', 'READ_TIMEOUT'),
        default=list(DEFAULT_REQUEST_TIMEOUT),
        help=f"Request timeout (connect, read) in seconds (default: {DEFAULT_REQUEST_TIMEOUT[0]} {DEFAULT_REQUEST_TIMEOUT[1]})"
    )
    parser.add_argument(
        '--max-retries', type=int, default=DEFAULT_MAX_RETRIES,
        help=f"Maximum number of retries for failed API requests (default: {DEFAULT_MAX_RETRIES})"
    )
    parser.add_argument(
        '--backoff-factor', type=float, default=DEFAULT_BACKOFF_FACTOR,
        help=f"Exponential backoff factor for retries (default: {DEFAULT_BACKOFF_FACTOR})"
    )
    parser.add_argument(
        '--max-consecutive-line-failures', type=int, default=DEFAULT_MAX_CONSECUTIVE_LINE_FAILURES,
        help=f"Maximum number of consecutive line processing failures within a single file before halting processing for that file (default: {DEFAULT_MAX_CONSECUTIVE_LINE_FAILURES}). Set to 0 to disable."
    )
    parser.add_argument(
        '--max-consecutive-file-failures', type=int, default=DEFAULT_MAX_CONSECUTIVE_FILE_FAILURES,
        help=f"Maximum number of consecutive files that fail processing before halting the entire script (default: {DEFAULT_MAX_CONSECUTIVE_FILE_FAILURES}). Set to 0 to disable."
    )

    return parser.parse_args()


def extract_doi_from_url(url_string):
    if not url_string or not isinstance(url_string, str):
        return None
    try:
        if url_string.lower().startswith("doi:"):
            doi_path = url_string[len("doi:"):].strip()
        else:
            parsed = urlparse(url_string)
            if parsed.netloc and parsed.netloc.lower() == 'doi.org':
                doi_path = parsed.path.lstrip('/')
            else:
                if url_string.strip().startswith("10."):
                    doi_path = url_string.strip()
                else:
                    logging.debug(f"URL '{url_string}' is not a doi.org URL and doesn't look like a DOI.")
                    return None
        doi_path = doi_path.strip()
        if doi_path:
            return doi_path
        else:
            return None
    except Exception as e:
        logging.warning(f"Could not parse URL/DOI string '{url_string}' to extract DOI: {e}")
    return None


def setup_logging(log_level_str, log_file=None):
    log_level_str_upper = log_level_str.upper()
    numeric_level = getattr(logging, log_level_str_upper, None)
    if not isinstance(numeric_level, int):
        print(f"Warning: Invalid log level '{log_level_str}'. Defaulting to INFO.", file=sys.stderr)
        numeric_level = logging.INFO
        log_level_str_upper = 'INFO'

    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(log_format, datefmt=date_format)

    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    if log_level_str_upper == 'NONE':
        root_logger.setLevel(logging.CRITICAL + 1)
        root_logger.addHandler(logging.NullHandler())
        print("Logging explicitly disabled ('NONE' selected).")
        return

    root_logger.setLevel(numeric_level)

    if log_file:
        try:
            handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
            print(f"Logging to file: {log_file} at level: {log_level_str_upper}")
        except IOError as e:
            print(f"Error opening log file {log_file}: {e}. Logging to stderr instead.", file=sys.stderr)
            handler = logging.StreamHandler(sys.stderr)
            print(f"Logging to stderr at level: {log_level_str_upper}")
    else:
        handler = logging.StreamHandler(sys.stderr)
        print(f"Logging to stderr at level: {log_level_str_upper}")

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    if root_logger.hasHandlers():
        logging.info(f"Logging configured successfully at level {log_level_str_upper}.")
    elif log_level_str_upper != 'NONE':
        print("Warning: Logging setup completed, but no handlers seem attached.", file=sys.stderr)


def process_single_file(input_file_path, output_file_path, matching_strategy, args):
    main_logger = logging.getLogger(__name__)
    main_logger.info(f"--- Starting processing of file: {input_file_path} ---")
    main_logger.info(f"Output format: {args.format.upper()}. Writing to: {output_file_path}")

    is_gzipped = input_file_path.lower().endswith('.gz')
    open_func = gzip.open if is_gzipped else open
    read_mode = 'rt'
    write_mode = 'wt'

    all_results_json = []
    processed_lines = 0
    matched_lines = 0
    lines_with_errors = 0
    consecutive_line_failures = 0
    output_writer = None
    file_halted_by_line_breaker = False

    try:
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

        with open_func(input_file_path, read_mode, encoding='utf-8') as infile, \
             open(output_file_path, write_mode, encoding='utf-8') as outfile:

            if args.format == 'csv':
                fieldnames = ['input_doi', 'matched_doi', 'confidence']
                output_writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                output_writer.writeheader()
                main_logger.debug(f"CSV writer initialized for {output_file_path}.")
            else:
                output_writer = all_results_json

            for i, line in enumerate(infile):
                line_num = i + 1
                line = line.strip()
                if not line:
                    main_logger.debug(f"Skipping empty line {line_num} in {input_file_path}.")
                    continue

                main_logger.debug(f"--- Processing Line {line_num} in {os.path.basename(input_file_path)} ---")
                input_doi_extracted = "N/A"
                matched_doi_extracted = None
                match_confidence_str = ""
                output_record = None
                line_processed_successfully = True

                try:
                    try:
                        input_json_string = line
                        input_data_parsed = json.loads(line)
                        if isinstance(input_data_parsed, dict):
                            input_doi_extracted = input_data_parsed.get('id') or input_data_parsed.get('doi')
                            if not input_doi_extracted:
                                main_logger.warning(f"Line {line_num} in {input_file_path}: Input JSON lacks 'id' or 'doi' field.")
                                input_doi_extracted = "N/A_MISSING_ID"
                            else:
                                input_doi_extracted = str(input_doi_extracted).replace("https://doi.org/", "").replace("doi:", "").strip()
                                main_logger.debug(f"Line {line_num}: Input DOI identified as '{input_doi_extracted}'.")
                        else:
                            main_logger.warning(f"Line {line_num} in {input_file_path}: Parsed JSON is not a dictionary (type: {type(input_data_parsed)}).")
                            input_doi_extracted = "N/A_INVALID_JSON_TYPE"
                    except json.JSONDecodeError as e:
                        main_logger.error(f"Line {line_num} in {input_file_path}: JSON decode error: {e}. Raw line (start): '{line[:100]}...'")
                        line_processed_successfully = False

                    if line_processed_successfully:
                        matches = matching_strategy.match(input_json_string)

                        if matches is None:
                            main_logger.error(f"Line {line_num} (Input DOI {input_doi_extracted}) in {input_file_path}: Strategy reported critical failure (likely API issue).")
                            line_processed_successfully = False
                        elif matches and isinstance(matches, list) and len(matches) > 0:
                            first_match = matches[0]
                            matched_doi_url = None
                            if isinstance(first_match, dict):
                                matched_doi_url = first_match.get('id')
                                confidence = first_match.get('confidence')

                                if isinstance(confidence, (int, float)):
                                    match_confidence_str = f"{confidence:.4f}"
                                elif confidence is not None:
                                    match_confidence_str = str(confidence)
                                else:
                                    match_confidence_str = ''

                                if matched_doi_url:
                                    matched_lines += 1
                                    main_logger.info(f"Line {line_num} (Input DOI {input_doi_extracted}): Found match '{matched_doi_url}' conf {match_confidence_str if match_confidence_str else 'N/A'}")
                                    matched_doi_extracted = extract_doi_from_url(matched_doi_url)
                                    if not matched_doi_extracted:
                                        main_logger.warning(f"Line {line_num} (Input DOI {input_doi_extracted}): Could not extract DOI from matched URL '{matched_doi_url}'.")
                                        match_confidence_str = ''
                                else:
                                    main_logger.warning(f"Line {line_num} (Input DOI {input_doi_extracted}): Match result dictionary lacks 'id' field: {first_match}")
                                    match_confidence_str = ''
                            else:
                                main_logger.warning(f"Line {line_num} (Input DOI {input_doi_extracted}): Match result item is not a dictionary: {type(first_match)}")
                                match_confidence_str = ''
                        else:
                            main_logger.info(f"Line {line_num} (Input DOI {input_doi_extracted}): No preprint match found.")
                            match_confidence_str = ''

                    if line_processed_successfully:
                        output_record = {
                            "input_doi": input_doi_extracted if not input_doi_extracted.startswith("N/A") else '',
                            "matched_doi": matched_doi_extracted if matched_doi_extracted else '',
                            "confidence": match_confidence_str if matched_doi_extracted else ''
                        }

                except Exception as e:
                    main_logger.error(f"Line {line_num} (Input DOI {input_doi_extracted}) in {input_file_path}: Unexpected error during processing: {e}", exc_info=True)
                    line_processed_successfully = False

                finally:
                    processed_lines += 1
                    if not line_processed_successfully:
                        lines_with_errors += 1
                        consecutive_line_failures += 1
                        main_logger.warning(f"Consecutive line failure count for {os.path.basename(input_file_path)}: {consecutive_line_failures}")
                    else:
                        if consecutive_line_failures > 0:
                            main_logger.info(f"Resetting consecutive line failure count from {consecutive_line_failures} after successful line {line_num} in {os.path.basename(input_file_path)}.")
                        consecutive_line_failures = 0

                    if args.max_consecutive_line_failures > 0 and consecutive_line_failures >= args.max_consecutive_line_failures:
                        main_logger.critical(f"Line-level circuit breaker tripped for file {input_file_path}: Reached {consecutive_line_failures} consecutive line failures "
                                            f"(threshold: {args.max_consecutive_line_failures}). Halting processing for this file.")
                        file_halted_by_line_breaker = True
                        break

                    if line_processed_successfully and output_record is not None:
                        if args.format == 'json':
                            output_writer.append(output_record)
                        elif args.format == 'csv' and output_writer:
                            try:
                                output_writer.writerow(output_record)
                            except Exception as e:
                                main_logger.error(f"Line {line_num}: Failed to write row to CSV for {output_file_path}: {e}. Data: {output_record}", exc_info=True)
                                lines_with_errors +=1
                                consecutive_line_failures +=1
                                main_logger.warning(f"Consecutive line failure count (CSV write error): {consecutive_line_failures}")
                                if args.max_consecutive_line_failures > 0 and consecutive_line_failures >= args.max_consecutive_line_failures:
                                    main_logger.critical(f"Line-level circuit breaker tripped after CSV write error for {input_file_path}. Halting.")
                                    file_halted_by_line_breaker = True
                                    break

                if processed_lines % 100 == 0 and not file_halted_by_line_breaker:
                    main_logger.info(f"Progress for {os.path.basename(input_file_path)}: Processed {processed_lines} lines... ({matched_lines} matched, {lines_with_errors} errors, {consecutive_line_failures} consecutive)")

            if args.format == 'json' and not file_halted_by_line_breaker:
                main_logger.info(f"Writing {len(all_results_json)} collected results as JSON to {output_file_path}...")
                try:
                    json.dump(all_results_json, outfile, ensure_ascii=False, indent=2)
                    main_logger.info("JSON writing complete.")
                except Exception as e:
                    main_logger.error(f"Failed to write JSON output to file '{output_file_path}': {e}", exc_info=True)
                    lines_with_errors += 1

    except FileNotFoundError:
        main_logger.error(f"Input file not found during processing: '{input_file_path}'")
        lines_with_errors = 1
    except gzip.BadGzipFile:
        main_logger.error(f"Input file '{input_file_path}' is corrupted or not a valid gzip file.")
        lines_with_errors = 1
    except IOError as e:
        main_logger.error(f"File I/O error accessing '{input_file_path}' or '{output_file_path}': {e}", exc_info=True)
        lines_with_errors = 1
    except Exception as e:
        main_logger.error(f"An unexpected error occurred processing file '{input_file_path}': {e}", exc_info=True)
        lines_with_errors = 1
    finally:
        main_logger.info(f"--- Finished processing file: {input_file_path} ---")
        if file_halted_by_line_breaker:
            main_logger.warning(f"Processing HALTED for this file due to line-level circuit breaker.")
        main_logger.info(f"Summary for {os.path.basename(input_file_path)}:")
        main_logger.info(f"  Total lines processed: {processed_lines}")
        main_logger.info(f"  Lines resulting in a match: {matched_lines}")
        if lines_with_errors > 0:
            main_logger.warning(f"  Total lines with errors (or file error like final write): {lines_with_errors}")
        else:
            main_logger.info("  No line processing errors or final write errors encountered for this file.")
        main_logger.info(f"  Results written to: {output_file_path} (Format: {args.format.upper()})")

    return lines_with_errors == 0 and not file_halted_by_line_breaker


def main():
    args = parse_arguments()
    setup_logging(args.log_level, args.log_file)
    main_logger = logging.getLogger(__name__)

    if not os.path.isdir(args.input):
        main_logger.critical(f"Fatal Error: Input path is not a valid directory: {args.input}")
        sys.exit(1)

    try:
        os.makedirs(args.output, exist_ok=True)
        main_logger.info(f"Output directory set to: {args.output}")
    except OSError as e:
        main_logger.critical(f"Fatal Error: Could not create output directory '{args.output}': {e}")
        sys.exit(1)

    try:
        request_timeout_tuple = tuple(args.timeout)
        matching_strategy = PreprintSbmvStrategy(
            mailto=args.mailto,
            user_agent=args.user_agent,
            min_score=args.min_score,
            max_score_diff=args.max_score_diff,
            weight_year=args.weight_year,
            weight_title=args.weight_title,
            weight_author=args.weight_author,
            max_query_len=args.max_query_len,
            request_timeout=request_timeout_tuple,
            max_retries=args.max_retries,
            backoff_factor=args.backoff_factor,
            logger_instance=logging.getLogger('strategy'),
            log_candidates=args.log_candidates,
            candidate_log_file=args.candidate_log_file
        )
        main_logger.info("Preprint matching strategy initialized successfully.")
    except Exception as e:
        main_logger.critical(f"Fatal Error: Could not initialize strategy: {e}", exc_info=True)
        sys.exit(1)

    files_to_process = []
    for root, _, files in os.walk(args.input):
        for filename in files:
            if filename.lower().endswith(".jsonl") or filename.lower().endswith(".jsonl.gz"):
                files_to_process.append(os.path.join(root, filename))

    if not files_to_process:
        main_logger.warning(f"No '.jsonl' or '.jsonl.gz' files found in input directory: {args.input}")
        sys.exit(0)

    main_logger.info(f"Found {len(files_to_process)} file(s) to process.")
    if args.max_consecutive_file_failures > 0:
        main_logger.info(f"File-level circuit breaker enabled: Halting script after {args.max_consecutive_file_failures} consecutive file processing failures.")
    else:
        main_logger.info("File-level circuit breaker disabled (max_consecutive_file_failures <= 0).")

    total_files_processed = 0
    total_files_failed_or_halted = 0
    consecutive_file_failures = 0

    for input_file_path in files_to_process:
        try:
            relative_path = os.path.relpath(input_file_path, args.input)
            base, ext = os.path.splitext(relative_path)
            if base.lower().endswith(".jsonl"):
                base, _ = os.path.splitext(base)
            
            output_filename = f"{base}.output.{args.format}"
            output_file_path = os.path.join(args.output, output_filename)

            success = process_single_file(input_file_path, output_file_path, matching_strategy, args)
            total_files_processed += 1

            if not success:
                total_files_failed_or_halted += 1
                consecutive_file_failures += 1
                main_logger.warning(f"Consecutive file failure count: {consecutive_file_failures}")
            else:
                if consecutive_file_failures > 0:
                    main_logger.info(f"Resetting consecutive file failure count from {consecutive_file_failures} after successful file {input_file_path}.")
                consecutive_file_failures = 0
            
            if args.max_consecutive_file_failures > 0 and consecutive_file_failures >= args.max_consecutive_file_failures:
                main_logger.critical(f"File-level circuit breaker tripped: Reached {consecutive_file_failures} consecutive file processing failures "
                                     f"(threshold: {args.max_consecutive_file_failures}). Halting script.")
                sys.exit(1)

        except SystemExit:
            raise
        except Exception as e:
            main_logger.critical(f"Fatal error during processing orchestration for {input_file_path}: {e}", exc_info=True)
            total_files_failed_or_halted += 1
            consecutive_file_failures += 1
            main_logger.critical("Attempting to continue with next file, but incrementing consecutive file failure count.")
            if args.max_consecutive_file_failures > 0 and consecutive_file_failures >= args.max_consecutive_file_failures:
                main_logger.critical(f"File-level circuit breaker tripped after orchestration error: Reached {consecutive_file_failures} consecutive failures. Halting script.")
                sys.exit(1)

    main_logger.info("--- Overall Processing Summary ---")
    main_logger.info(f"Total files found: {len(files_to_process)}")
    main_logger.info(f"Total files attempted processing: {total_files_processed}")
    if total_files_failed_or_halted > 0:
        main_logger.warning(f"Total files with errors or halted by line/file circuit breakers: {total_files_failed_or_halted}")
    else:
        main_logger.info("All attempted files processed without critical errors or halts.")
    main_logger.info(f"Final consecutive file failure count: {consecutive_file_failures}")
    main_logger.info("--- Script Finished ---")

    if 'matching_strategy' in locals() and hasattr(matching_strategy, 'session') and matching_strategy.session:
        matching_strategy.session.close()
        main_logger.debug("Closed strategy session at script end.")


if __name__ == "__main__":
    main()