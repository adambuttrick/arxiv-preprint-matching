# arXiv Preprint Matching

Modified form of [Search Based Matching with Validation (SBMV) preprint matching strategy](https://gitlab.com/crossref/labs/marple/-/blob/main/strategies_available/preprint_sbmv/strategy.py?ref_type=heads) developed by [@dtkaczyk](https://github.com/dtkaczyk), specifically adapted for matching arXiv preprint DOIs represented in the DataCite schema.


## Installation

```bash
pip install -r requirements.txt
```

## Usage

The script processes all `.jsonl` and `.jsonl.gz` files found within a specified input directory, saving results to a corresponding structure in an output directory.

```bash
python preprint_match_data_files.py -i INPUT_DIR -f FORMAT -m EMAIL -u USER_AGENT [OPTIONS]
```

### Required Arguments
- `-i, --input`: Path to the input directory containing .jsonl or .jsonl.gz files.
- `-f, --format`: Output format ('json' or 'csv') for the result files.
- `-m, --mailto`: Email address for Crossref API politeness (required by Crossref).
- `-u, --user-agent`: User-Agent string for API requests (e.g., "arXivPreprintMatcher/1.0").

### Optional Arguments
#### Input/Output:
- `-o, --output`: Path to the output directory where results will be saved. Will be created if it doesn't exist (default: `./output`). Output files mirror the input structure.

#### Logging:
- `-ll, --log-level`: Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL, NONE). Default: INFO.
- `-lf, --log-file`: Path to log file (defaults to stderr).
- `-lc, --log-candidates`: If set, logs raw Crossref candidate results (appended for all files processed).
- `-cf, --candidate-log-file`: Path for logging candidates (default: crossref_candidates.log).

#### Strategy Parameters:
- `--min-score`: Minimum score threshold for a match (default: 0.85).
- `--max-score-diff`: Maximum allowed difference from top score for multiple matches (default: 0.03).
- `--weight-year`: Weight for the year score component (default: 0.4).
- `--weight-title`: Weight for the title score component (default: 2.0).
- `--weight-author`: Weight for the author score component (default: 0.8).
- `--max-query-len`: Maximum length of the query string sent to Crossref (default: 5000).

#### File Processing & API Handling:
- `--timeout`: Request timeout (connect, read) in seconds (default: 10 30).
- `--max-retries`: Maximum number of retries for failed API requests (default: 3).
- `--backoff-factor`: Exponential backoff factor for retries (default: 0.5).
- `--max-consecutive-line-failures`: Maximum number of consecutive lines that fail (due to JSON errors, persistent API errors after retries, or other processing exceptions) within a single file before halting processing *for that file* (default: 10). Set to 0 to disable.
- `--max-consecutive-file-failures`: Maximum number of consecutive files that fail processing (due to file errors or being halted by the line-level breaker) before halting the *entire script* (default: 3). Set to 0 to disable.

## Examples

Process all `.jsonl`/`.jsonl.gz` files in `input_data/` and save CSV results to `output_results/` (creating it if needed):

```bash
python preprint_match_data_files.py -i input_data/ -o output_results/ -f csv -m <your_email@example.com> -u "MyMatchingTool/1.1 (mailto:your_email@example.com)"
```

Process files in `preprints/`, saving JSON results to the default `./output` directory, with detailed logging and custom API/strategy settings:

```bash
python preprint_match_data_files.py -i preprints/ -f json -m <your_email@example.com> -u "arXivPreprintMatcher/1.0" \
 -ll DEBUG -lf matching.log -lc \
 --min-score 0.8 --weight-title 1.5 --weight-author 1.0 \
 --timeout 15 45 --max-retries 5 --max-consecutive-failures 20
```

## Description of Strategy

This strategy attempts to find potential published versions (primarily journal articles) corresponding to input preprint records (expected in DataCite JSON format). It uses the Crossref API and applies scoring based on metadata similarity.


### Search Approach and Candidate Filtering

1.  A bibliographic query string is built using metadata extracted from the DataCite input: the main title (and subtitle, if present), publication year, and the family names of personal authors listed as `creators` or `contributors`. These components are normalized using `unidecode`, lowercasing, and removing punctuation before constructing the query.
2.  The query targets the Crossref `/works` endpoint via a robust HTTP session, using the `query.bibliographic` parameter and returning up to 25 candidates (`rows=25`). The maximum query length is capped (default 5000, adjustable via `--max-query-len`).
3.  Candidates retrieved from Crossref are filtered based on their work type (`type` field) to include only relevant publication types such as `journal-article`, `proceedings-article`, `book-chapter`, `report`, and `posted-content`.

### Scoring Logic, Weights, and Heuristics:

The strategy employs weighted scoring based on year, title, and author similarity, incorporating fuzzy matching and heuristics:

* **Year Score:**
    * Compares the preprint's `publicationYear` with the candidate's publication year (extracted carefully from fields like `published-online`, `published-print`, `issued`, `created`).
    * Assigns scores based on the difference (`candidate_year - preprint_year`): 1.0 for diff 0-2; 0.9 for diff 3; 0.8 for diff 4; 0.0 otherwise (penalizing cases where candidate significantly predates or postdates the preprint). Returns 0.0 if years cannot be compared.
* **Title Score:**
    * Compares normalized titles (input vs. candidate). Normalization includes Unicode handling, accent removal, lowercasing, and punctuation stripping.
    * Uses a weighted blend of fuzzy matching scores: `0.4 * fuzz.token_set_ratio + 0.4 * fuzz.token_sort_ratio + 0.2 * fuzz.WRatio`.
    * Applies a penalty (`*= 0.7`) if the *first normalized word* of one title contains keywords like "correction", "reply", "erratum", etc., while the other title does not.
* **Author Score:**
    * Applies several heuristics for comparing normalized author lists:
        * An exact match between valid, normalized ORCIDs results in a score of 1.0; a mismatch results in 0.0, skipping name comparison.
        * Iteratively finds the most similar pair of authors between the two lists using `_score_normalized_author_similarity`. This comparison uses `fuzz.token_sort_ratio` on pre-calculated, normalized name variations (e.g., "J Smith", "Smith J", "John Smith", "Smith John").
        * Author pairs with a similarity below 0.5 are discarded during the greedy matching.
        * If family names match *and* the name similarity score is > 0.6, the pair's score is boosted slightly (`* 1.1`).
        * For efficiency, compares sorted strings of normalized family names using `fuzz.token_sort_ratio`.
        * The final author score (based on the sum of matched pair scores in the greedy approach) is normalized by the total number of unique authors involved: `(2.0 * score_sum) / total_authors`, clamped between 0.0 and 1.0. Handles empty lists gracefully.
* **Final Weighted Score:** Calculated as: `(weight_year * year_score + weight_title * title_score + weight_author * author_score) / (weight_year + weight_title + weight_author)`. Default weights are `weight_title=2.0`, `weight_author=0.8`, `weight_year=0.4`. These weights can be adjusted via command-line arguments.

### Match Selection

1.  Only candidates achieving a final weighted score >= `min_score` (default 0.85) are considered potential matches.
2.  Among these, only candidates whose scores are within `max_score_diff` (default 0.03) of the *highest* score obtained for that input record are returned as the final match(es). This helps select the best result(s) when multiple candidates have very similar high scores.

