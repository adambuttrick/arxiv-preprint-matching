import re
import json
import logging
import datetime
import unicodedata
import requests
from rapidfuzz import fuzz
from unidecode import unidecode

try:
    from pylate import models, rank
    import torch
    RERANKER_AVAILABLE = True
except ImportError as e:
    RERANKER_AVAILABLE = False
    _reranker_import_error = str(e)

from matching.utils import (
    crossref_rest_api_call,
    doi_id,
    get_crossref_api_session,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_MAX_RETRIES,
    DEFAULT_BACKOFF_FACTOR,
    DEFAULT_STATUS_FORCELIST
)


class PreprintSbmvStrategy:
    strategy = "preprint-sbmv-datacite"
    task = "preprint-matching"
    description = (
        "This strategy uses Crossref's REST API to search for candidate "
        + "preprints based on metadata (title, creators, contributors, year) "
        + "extracted from a DataCite JSON input, and compares the normalized metadata "
        + "to validate the candidates using enhanced scoring."
    )
    default = False

    DEFAULT_MIN_SCORE = 0.8
    DEFAULT_MAX_SCORE_DIFF = 0.03
    DEFAULT_MAX_QUERY_LEN = 5000
    DEFAULT_WEIGHT_YEAR = 0.4
    DEFAULT_WEIGHT_TITLE = 2.0
    DEFAULT_WEIGHT_AUTHOR = 1.0

    accepted_crossref_types = [
        "journal-article",
        "proceedings-article",
        "book-chapter",
        "report",
        "posted-content"
    ]

    def __init__(self, mailto, user_agent,
                 min_score=DEFAULT_MIN_SCORE,
                 max_score_diff=DEFAULT_MAX_SCORE_DIFF,
                 weight_year=DEFAULT_WEIGHT_YEAR,
                 weight_title=DEFAULT_WEIGHT_TITLE,
                 weight_author=DEFAULT_WEIGHT_AUTHOR,
                 max_query_len=DEFAULT_MAX_QUERY_LEN,
                 enable_reranker=False,
                 reranker_model_path='lightonai/GTE-ModernColBERT-v1',
                 reranker_batch_size=16,
                 heuristic_weight=0.3,
                 reranker_weight=0.6,
                 request_timeout=DEFAULT_REQUEST_TIMEOUT,
                 max_retries=DEFAULT_MAX_RETRIES,
                 backoff_factor=DEFAULT_BACKOFF_FACTOR,
                 status_forcelist=DEFAULT_STATUS_FORCELIST,
                 logger_instance=None,
                 log_candidates=False,
                 candidate_log_file="crossref_candidates.log"):

        if not mailto or not user_agent:
            raise ValueError(
                "mailto and user_agent are required for Strategy initialization.")
        self.mailto = mailto
        self.user_agent = user_agent
        self.logger = logger_instance if logger_instance else logging.getLogger(
            __name__)
        if not logger_instance and not logging.getLogger().hasHandlers():
            logging.basicConfig(
                level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

        self.min_score = float(min_score)
        self.max_score_diff = float(max_score_diff)
        self.weight_year = float(weight_year)
        self.weight_title = float(weight_title)
        self.weight_author = float(weight_author)
        self.max_query_len = int(max_query_len)

        self.request_timeout = request_timeout
        self.max_retries = int(max_retries)
        self.backoff_factor = float(backoff_factor)
        self.status_forcelist = tuple(status_forcelist)

        self.log_candidates = log_candidates
        self.candidate_log_file = candidate_log_file
        if self.log_candidates:
            self.logger.info(f"Candidate logging enabled. Raw candidates will be saved to: {self.candidate_log_file}")

        self.enable_reranker = enable_reranker
        self.reranker_model = None
        self.reranker_batch_size = reranker_batch_size
        self.heuristic_weight = heuristic_weight
        self.reranker_weight = reranker_weight

        if self.enable_reranker:
            if not RERANKER_AVAILABLE:
                self.logger.error(f"Reranker dependencies not available: {_reranker_import_error}")
                self.enable_reranker = False
                self.logger.warning("Reranker disabled due to missing dependencies. Falling back to heuristic-only mode.")
            else:
                try:
                    self.logger.info(f"Reranker enabled. Loading model: {reranker_model_path}...")
                    self.reranker_model = models.ColBERT(model_name_or_path=reranker_model_path)
                    self.logger.info("Reranker model loaded successfully.")
                except Exception as e:
                    self.logger.error(f"Failed to load reranker model: {e}", exc_info=True)
                    self.enable_reranker = False
                    self.logger.warning("Reranker disabled due to loading failure. Falling back to heuristic-only mode.")

        try:
            self.session = get_crossref_api_session(
                max_retries=self.max_retries,
                backoff_factor=self.backoff_factor,
                status_forcelist=self.status_forcelist
            )
            self.logger.info("Requests session with retry logic initialized.")
        except Exception as e:
            self.logger.error(f"Failed to initialize requests session: {e}", exc_info=True)
            self.session = None

        self.logger.info(f"Strategy initialized with parameters: min_score={self.min_score}, "
                         f"max_score_diff={self.max_score_diff}, weight_year={self.weight_year}, "
                         f"weight_title={self.weight_title}, weight_author={self.weight_author}, "
                         f"timeout={self.request_timeout}, max_retries={self.max_retries}, "
                         f"enable_reranker={self.enable_reranker}, reranker_batch_size={self.reranker_batch_size}, "
                         f"heuristic_weight={self.heuristic_weight}, reranker_weight={self.reranker_weight}")

    def _get_text_from_datacite(self, article_datacite):
        attributes = article_datacite.get("attributes", {})
        if not isinstance(attributes, dict):
            return ""
        title = ""
        titles_list = attributes.get("titles", [])
        main_title = ""
        subtitle = ""
        if isinstance(titles_list, list):
            found_main = False
            for t in titles_list:
                if isinstance(t, dict):
                    t_type = t.get("titleType", "").lower() if t.get(
                        "titleType") else "main"
                    t_title = t.get("title", "")
                    if t_title:
                        if t_type == "main" and not main_title:
                            main_title = t_title
                            found_main = True
                        elif t_type == "subtitle" and not subtitle:
                            subtitle = t_title
            if not found_main and not main_title and titles_list:
                for t in titles_list:
                    if isinstance(t, dict) and t.get("title"):
                        main_title = t.get("title")
                        break

        if main_title:
            title = main_title
            if subtitle:
                title = f"{main_title}: {subtitle}".strip().rstrip(':').strip()

        if title:
            title = re.sub("&amp;", "&", title)
            title = re.sub("&lt;", "<", title)
            title = re.sub("&gt;", ">", title)
            title = re.sub(r'\s+', ' ', title).strip()

        author_names = []
        people = []
        creators = attributes.get("creators", [])
        contributors = attributes.get("contributors", [])
        if isinstance(creators, list):
            people.extend(creators)
        if isinstance(contributors, list):
            people.extend([p for p in contributors if isinstance(
                p, dict) and p.get("nameType") == "Personal"])

        for person in people:
            if isinstance(person, dict) and person.get("nameType") == "Personal":
                family_name = person.get("familyName")
                if family_name and family_name.strip():
                    author_names.append(family_name.strip())

        authors_str = " ".join(sorted(list(set(filter(None, author_names)))))

        return f"{title} {authors_str}".strip()

    def _get_text_from_crossref(self, candidate_crossref):
        if not isinstance(candidate_crossref, dict):
            return ""

        title = ""
        cr_titles = candidate_crossref.get("title", [])
        if cr_titles and isinstance(cr_titles, list):
            title = cr_titles[0] if cr_titles[0] else ""

        authors = []
        cr_authors = candidate_crossref.get("author", [])
        if cr_authors and isinstance(cr_authors, list):
            for author in cr_authors:
                if isinstance(author, dict) and author.get("family"):
                    authors.append(author.get("family"))

        return f"{title} {' '.join(authors)}".strip()

    def __del__(self):
        if hasattr(self, 'session') and self.session:
            self.session.close()
            self.logger.debug("Closed strategy requests session.")

    def _normalize_string(self, text):
        if text is None:
            return ""
        try:
            text_str = str(text)
            normalized_text = unicodedata.normalize('NFC', text_str)
            normalized_text = unidecode(normalized_text)
            normalized_text = normalized_text.lower()
            normalized_text = re.sub(
                r'[\u2010\u2011\u2012\u2013\u2014\u2015]', '-', normalized_text)
            normalized_text = re.sub(
                r'[^\w\s-]', '', normalized_text, flags=re.UNICODE)
            normalized_text = re.sub(r'\s+', ' ', normalized_text).strip()
            return normalized_text
        except Exception as e:
            self.logger.error(f"Error normalizing string '{str(text)[:100]}...': {e}", exc_info=False)
            temp_text = re.sub(r'\s+', ' ', str(text)).strip().lower()
            temp_text = re.sub(r'[^\w\s-]', '', temp_text, flags=re.UNICODE)
            return temp_text

    def match(self, input_json_string):
        try:
            article_datacite = json.loads(input_json_string)
            input_doi = article_datacite.get(
                "id") or article_datacite.get("doi", "N/A")
            if not isinstance(article_datacite, dict):
                self.logger.error(f"input JSON for DOI '{input_doi}' did not parse into a dictionary.")
                return None
        except json.JSONDecodeError as e:
            self.logger.error(f"Error decoding input JSON: {e}. input (start): '{input_json_string[:200]}...'")
            return None
        except Exception as e:
            self.logger.error(f"Error processing input JSON structure: {e}")
            return None

        query = self.candidate_query(article_datacite)
        if not query:
            return []

        candidates_crossref = self.get_candidates(article_datacite, query)

        if candidates_crossref is None:
            self.logger.warning(f"Candidate retrieval failed for input DOI {input_doi}, cannot proceed with matching.")
            return None

        filtered_candidates = [
            r for r in candidates_crossref
            if isinstance(r, dict) and r.get("type", "").lower() in self.accepted_crossref_types
        ]
        if len(filtered_candidates) < len(candidates_crossref):
            self.logger.debug(f"Filtered {len(candidates_crossref) - len(filtered_candidates)} candidates by type for input DOI {input_doi}.")

        matches = self.match_candidates(article_datacite, filtered_candidates)
        return matches

    def get_candidates(self, article_datacite, query):
        input_doi = article_datacite.get(
            "id") or article_datacite.get("doi", "N/A")

        if not self.session:
            self.logger.error(f"Cannot get candidates for {input_doi}: Requests session not initialized.")
            return None

        params = {
            "query.bibliographic": query,
            "rows": 25
        }

        code, results = crossref_rest_api_call(
            route="works",
            params=params,
            mailto=self.mailto,
            user_agent=self.user_agent,
            session=self.session,
            timeout=self.request_timeout
        )

        if self.log_candidates:
            self._log_raw_candidates(input_doi, query, code, results)

        if code is None:
            self.logger.error(f"Crossref API call failed completely for input DOI {input_doi} after retries. Error details: {results}")
            return None
        elif code != 200:
            self.logger.warning(f"Crossref API call for input DOI {input_doi} returned status {code}. Details: {str(results)[:300]}")
            return []
        elif not isinstance(results, dict) or "items" not in results:
            self.logger.warning(f"Unexpected Crossref API response structure (Status 200) for input DOI {input_doi}. Response type: {type(results)}, Content (start): {str(results)[:200]}")
            return []
        else:
            items = results.get("items", [])
            if not isinstance(items, list):
                self.logger.warning(f"Crossref API 'items' is not a list for input DOI {input_doi}. Found type: {type(items)}")
                return []
            self.logger.debug(f"Retrieved {len(items)} raw candidates for input DOI {input_doi}.")
            return items

    def _log_raw_candidates(self, input_doi, query, code, results):
        try:
            log_entry = {
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                "input_doi": input_doi,
                "query": query,
                "api_status_code": code if code is not None else "N/A (Exception)",
                "retrieved_candidates": []
            }
            if code == 200 and isinstance(results, dict) and "items" in results:
                log_entry["retrieved_candidates"] = results.get("items", [])
            elif code is not None:
                log_entry["error_details"] = str(results)[:1000]
            else:
                log_entry["error_details"] = str(results)[:1000]

            with open(self.candidate_log_file, 'a', encoding='utf-8') as f:
                json.dump(log_entry, f, ensure_ascii=False)
                f.write('\n')
        except IOError as e:
            self.logger.error(f"Failed to write to candidate log file {self.candidate_log_file}: {e}", exc_info=True)
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during candidate logging: {e}", exc_info=True)

    def match_candidates(self, article_datacite, candidates_crossref):
        input_doi = article_datacite.get("id") or article_datacite.get("doi", "N/A")
        if not candidates_crossref:
            return []
        heuristic_scores = {}
        for cand in candidates_crossref:
            if isinstance(cand, dict) and cand.get("DOI"):
                cand_doi = cand.get("DOI")
                score = self.score(article_datacite, cand)
                if score is not None:
                    heuristic_scores[cand_doi] = score
            else:
                if isinstance(cand, dict) and "DOI" not in cand:
                    self.logger.debug(f"Candidate missing DOI field for input DOI {input_doi}: {str(cand)[:100]}")
                elif not isinstance(cand, dict):
                    self.logger.debug(f"Candidate item is not a dict for input DOI {input_doi}: {str(cand)[:100]}")

        if not self.enable_reranker or not self.reranker_model or not heuristic_scores:
            matches = [(d, s) for d, s in heuristic_scores.items() if s >= self.min_score]
            
            if not matches:
                self.logger.debug(f"No candidates met min_score ({self.min_score}) for input DOI {input_doi}.")
                return []

            try:
                top_score = max((s for _, s in matches), default=None)
                if top_score is None:
                    self.logger.warning(f"Could not determine top score for DOI {input_doi} despite having matches.")
                    return []
            except ValueError:
                self.logger.warning(f"Value error determining top score for DOI {input_doi}.")
                return []

            final_matches = [(d, s) for d, s in matches if top_score - s < self.max_score_diff]

            formatted_results = [
                {"id": doi_id(doi), "confidence": round(score, 4), "strategies": [self.strategy]}
                for doi, score in final_matches
            ]

            if formatted_results:
                self.logger.info(f"Found {len(formatted_results)} final match(es) for input DOI {input_doi} "
                                f"(Top score: {top_score:.4f}, Min score: {self.min_score}, Max diff: {self.max_score_diff}).")
            return formatted_results

        reranker_scores = {}
        try:
            query_text = self._get_text_from_datacite(article_datacite)
            
            docs_to_rerank = []
            doc_ids_to_rerank = []
            for cand in candidates_crossref:
                cand_doi = cand.get("DOI")
                if cand_doi in heuristic_scores:
                    docs_to_rerank.append(self._get_text_from_crossref(cand))
                    doc_ids_to_rerank.append(cand_doi)
            
            if query_text and docs_to_rerank:
                self.logger.debug(f"Reranking {len(docs_to_rerank)} candidates for input DOI {input_doi}")
                
                query_embedding = self.reranker_model.encode([query_text], is_query=True, batch_size=1)
                doc_embeddings = self.reranker_model.encode(docs_to_rerank, is_query=False, batch_size=self.reranker_batch_size)

                reranked_results = rank.rerank(
                    documents_ids=[doc_ids_to_rerank],
                    queries_embeddings=query_embedding,
                    documents_embeddings=[doc_embeddings]
                )

                if reranked_results and reranked_results[0]:
                    scores_list = [item['score'] for item in reranked_results[0]]
                    min_score, max_score = min(scores_list), max(scores_list)
                    
                    for item in reranked_results[0]:
                        normalized_score = (item['score'] - min_score) / (max_score - min_score) if max_score > min_score else 0.0
                        reranker_scores[item['id']] = normalized_score
        
        except Exception as e:
            self.logger.error(f"Reranking failed for input DOI {input_doi}: {e}", exc_info=True)
            reranker_scores = {}

        final_scores = {}
        for doi, h_score in heuristic_scores.items():
            r_score = reranker_scores.get(doi, 0.0)
            
            if reranker_scores:
                final_score = (self.heuristic_weight * h_score) + (self.reranker_weight * r_score)
            else:
                final_score = h_score
                
            final_scores[doi] = final_score
            self.logger.debug(f"Scores for {doi}: Heuristic={h_score:.4f}, Reranker={r_score:.4f} -> Final={final_score:.4f}")

        matches = [(d, s) for d, s in final_scores.items() if s >= self.min_score]
        
        if not matches:
            self.logger.debug(f"No candidates met min_score ({self.min_score}) for input DOI {input_doi}.")
            return []

        top_score = max((s for _, s in matches), default=0.0)
        final_matches = [(d, s) for d, s in matches if top_score - s < self.max_score_diff]

        formatted_results = [
            {"id": doi_id(doi), "confidence": round(score, 4), "strategies": [self.strategy]}
            for doi, score in final_matches
        ]
        
        if formatted_results:
            score_type = "hybrid" if reranker_scores else "heuristic-only"
            self.logger.info(f"Found {len(formatted_results)} final match(es) for input DOI {input_doi} using {score_type} scoring "
                            f"(Top score: {top_score:.4f}, Min score: {self.min_score}, Max diff: {self.max_score_diff}).")

        return formatted_results

    def candidate_query(self, article_datacite):
        input_doi = article_datacite.get(
            "id") or article_datacite.get("doi", "N/A")
        attributes = article_datacite.get("attributes", {})
        if not isinstance(attributes, dict):
            self.logger.warning(f"input attributes missing or not a dict for DOI {input_doi}. Cannot generate query.")
            return ""

        title = ""
        titles_list = attributes.get("titles", [])
        main_title = ""
        subtitle = ""
        if isinstance(titles_list, list):
            found_main = False
            for t in titles_list:
                if isinstance(t, dict):
                    t_type = t.get("titleType", "").lower() if t.get(
                        "titleType") else "main"
                    t_title = t.get("title", "")
                    if t_title:
                        if t_type == "main" and not main_title:
                            main_title = t_title
                            found_main = True
                        elif t_type == "subtitle" and not subtitle:
                            subtitle = t_title
            if not found_main and not main_title and titles_list:
                for t in titles_list:
                    if isinstance(t, dict) and t.get("title"):
                        main_title = t.get("title")
                        break

        if main_title:
            title = main_title
            if subtitle:
                title = f"{main_title}: {subtitle}".strip().rstrip(':').strip()

        if title:
            title = re.sub("&amp;", "&", title)
            title = re.sub("&lt;", "<", title)
            title = re.sub("&gt;", ">", title)
            title = re.sub(r'\s+', ' ', title).strip()

        author_names = []
        people = []
        creators = attributes.get("creators", [])
        contributors = attributes.get("contributors", [])
        if isinstance(creators, list):
            people.extend(creators)
        if isinstance(contributors, list):
            people.extend([p for p in contributors if isinstance(
                p, dict) and p.get("nameType") == "Personal"])

        for person in people:
            if isinstance(person, dict) and person.get("nameType") == "Personal":
                family_name = person.get("familyName")
                if family_name and family_name.strip():
                    author_names.append(family_name.strip())

        authors_str = " ".join(sorted(list(set(filter(None, author_names)))))

        year_str = ""
        pub_year = attributes.get("publicationYear")
        if pub_year:
            try:
                year_val = int(pub_year)
                if 1800 < year_val < 2100:
                    year_str = str(year_val)
                else:
                    self.logger.debug(f"Publication year {year_val} outside expected range (1800-2100) for DOI {input_doi}")
            except (ValueError, TypeError):
                self.logger.warning(f"Invalid publicationYear format '{pub_year}' for DOI {input_doi}. Skipping year in query.")
                year_str = ""

        query_parts = [part for part in [title, authors_str, year_str] if part]
        full_query = " ".join(query_parts)
        full_query = re.sub(r'\s+', ' ', full_query).strip()

        if not title and not authors_str:
            self.logger.warning(f"Query for DOI {input_doi} lacks both title and author information. Skipping API call.")
            return ""

        if not full_query:
            self.logger.warning(f"Empty query generated for DOI {input_doi} despite having title/authors. Skipping.")
            return ""

        final_query = full_query[:self.max_query_len]
        if len(full_query) > self.max_query_len:
            self.logger.warning(f"Query truncated for DOI {input_doi}. Original length: {len(full_query)}, Max length: {self.max_query_len}")

        self.logger.debug(f"Generated Query for {input_doi}: '{final_query[:200]}...'")
        return final_query

    def score(self, article_datacite, preprint_crossref):
        input_doi = article_datacite.get(
            "id") or article_datacite.get("doi", "N/A")
        preprint_doi = preprint_crossref.get(
            "DOI", "N/A") if isinstance(preprint_crossref, dict) else "N/A"

        if not isinstance(preprint_crossref, dict):
            self.logger.warning(f"Candidate {preprint_doi} is not a dictionary during scoring for input DOI {input_doi}.")
            return None

        y_score = self.year_score(article_datacite, preprint_crossref)
        t_score = self.title_score(article_datacite, preprint_crossref)
        a_score = self.authors_score(article_datacite, preprint_crossref)

        if y_score is None or t_score is None or a_score is None:
            self.logger.warning(f"Failed to calculate one or more score components for input:{input_doi} vs Cand:{preprint_doi}. Cannot compute final score.")
            return None

        weighted_sum = (self.weight_year * y_score +
                        self.weight_title * t_score +
                        self.weight_author * a_score)

        total_weight = (self.weight_year +
                        self.weight_title +
                        self.weight_author)

        if total_weight == 0:
            final_score = 0.0
        else:
            final_score = weighted_sum / total_weight
            final_score = max(0.0, min(1.0, final_score))

        self.logger.debug(f"Scores for input:{input_doi} vs Cand:{preprint_doi}: "
                          f"Year={y_score:.3f} (w={self.weight_year}), "
                          f"Title={t_score:.3f} (w={self.weight_title}), "
                          f"Author={a_score:.3f} (w={self.weight_author}) -> Final={final_score:.4f}")
        return final_score

    def year_score(self, article_datacite, preprint_crossref):
        input_doi = article_datacite.get(
            "id") or article_datacite.get("doi", "N/A")
        preprint_doi = preprint_crossref.get("DOI", "N/A")

        try:
            article_year = None
            attributes = article_datacite.get("attributes", {})
            if isinstance(attributes, dict):
                pub_year = attributes.get("publicationYear")
                if pub_year is not None:
                    try:
                        year_val = int(pub_year)
                        if 1800 < year_val < 2100:
                            article_year = year_val
                        else:
                            self.logger.debug(f"input Year {year_val} out of range for {input_doi}")
                    except (ValueError, TypeError):
                        self.logger.debug(f"Invalid publicationYear '{pub_year}' in DataCite for {input_doi}")

            preprint_year = None
            date_sources = [
                preprint_crossref.get("published-online"),
                preprint_crossref.get("published-print"),
                preprint_crossref.get("issued"),
                preprint_crossref.get("created"),
            ]

            for source_name, date_info in zip(["published-online", "published-print", "issued", "created"], date_sources):
                if isinstance(date_info, dict) and date_info.get("date-parts"):
                    date_parts = date_info["date-parts"]
                    if (isinstance(date_parts, list) and date_parts and
                            isinstance(date_parts[0], list) and date_parts[0]):
                        year_part = date_parts[0][0]
                        if year_part is not None:
                            try:
                                potential_year = int(year_part)
                                if 1800 < potential_year < 2100:
                                    preprint_year = potential_year
                                    self.logger.debug(f"Extracted CandYear={preprint_year} from '{source_name}' for {preprint_doi}")
                                    break
                                else:
                                    self.logger.debug(f"Cand Year {potential_year} from '{source_name}' out of range for {preprint_doi}")
                            except (ValueError, TypeError):
                                self.logger.debug(f"Non-integer year part '{year_part}' from '{source_name}' for {preprint_doi}")
                                continue
            if article_year is None or preprint_year is None:
                if article_year is not None or preprint_year is not None:
                    self.logger.debug(f"Missing year for comparison: inputYear={article_year}, CandYear={preprint_year}. input:{input_doi}, Cand:{preprint_doi}. Returning score 0.0.")
                return 0.0

            year_diff = preprint_year - article_year

            if year_diff < 0:
                score = 0.0
            elif 0 <= year_diff <= 2:
                score = 1.0
            elif year_diff == 3:
                score = 0.9
            elif year_diff == 4:
                score = 0.8
            else:
                score = 0.0

            return score

        except (AttributeError, TypeError, IndexError, ValueError) as e:
            self.logger.error(f"Error calculating year score for input:{input_doi} vs Cand:{preprint_doi}: {e}", exc_info=False)
            return None

    def title_score(self, article_datacite, preprint_crossref):
        input_doi = article_datacite.get(
            "id") or article_datacite.get("doi", "N/A")
        preprint_doi = preprint_crossref.get("DOI", "N/A")

        try:
            article_title_raw = ""
            attributes = article_datacite.get("attributes", {})
            if isinstance(attributes, dict):
                titles_list = attributes.get("titles", [])
                main_title = ""
                subtitle = ""
                if isinstance(titles_list, list):
                    found_main = False
                    for t in titles_list:
                        if isinstance(t, dict):
                            t_type = t.get("titleType", "").lower() if t.get(
                                "titleType") else "main"
                            t_title = t.get("title", "")
                            if t_title:
                                if t_type == "main" and not main_title:
                                    main_title = t_title
                                    found_main = True
                                elif t_type == "subtitle" and not subtitle:
                                    subtitle = t_title
                    if not found_main and not main_title and titles_list:
                        for t in titles_list:
                            if isinstance(t, dict) and t.get("title"):
                                main_title = t.get("title")
                                break
                if main_title:
                    article_title_raw = main_title
                    if subtitle:
                        article_title_raw = f"{main_title}: {subtitle}".strip().rstrip(':').strip()
            article_title_raw = re.sub("&amp;", "&", article_title_raw)
            article_title_raw = re.sub("&lt;", "<", article_title_raw)
            article_title_raw = re.sub("&gt;", ">", article_title_raw)
            article_title_raw = re.sub(r'\s+', ' ', article_title_raw).strip()

            preprint_title_raw = ""
            crossref_titles = preprint_crossref.get("title", [])
            if isinstance(crossref_titles, list) and crossref_titles:
                first_cr_title = crossref_titles[0]
                if isinstance(first_cr_title, str):
                    preprint_title_raw = first_cr_title

            crossref_subtitles = preprint_crossref.get("subtitle", [])
            if preprint_title_raw and isinstance(crossref_subtitles, list) and crossref_subtitles:
                first_cr_subtitle = crossref_subtitles[0]
                if isinstance(first_cr_subtitle, str) and first_cr_subtitle.strip():
                    preprint_title_raw = f"{preprint_title_raw}: {first_cr_subtitle}".strip().rstrip(':').strip()
            preprint_title_raw = re.sub("&amp;", "&", preprint_title_raw)
            preprint_title_raw = re.sub("&lt;", "<", preprint_title_raw)
            preprint_title_raw = re.sub("&gt;", ">", preprint_title_raw)
            preprint_title_raw = re.sub(
                r'\s+', ' ', preprint_title_raw).strip()

            article_title_norm = self._normalize_string(article_title_raw)
            preprint_title_norm = self._normalize_string(preprint_title_raw)

            if not article_title_norm or not preprint_title_norm:
                if article_title_norm or preprint_title_norm:
                    self.logger.debug(f"Missing normalized title for comparison: input:'{article_title_norm}' vs Cand:'{preprint_title_norm}' (input DOI:{input_doi}, Cand DOI:{preprint_doi}). Score 0.0")
                return 0.0

            score_ts = fuzz.token_set_ratio(
                article_title_norm, preprint_title_norm) / 100.0
            score_tso = fuzz.token_sort_ratio(
                article_title_norm, preprint_title_norm) / 100.0
            score_w = fuzz.WRatio(article_title_norm,
                                  preprint_title_norm) / 100.0

            score = (score_ts * 0.4 + score_tso * 0.4 + score_w * 0.2)

            def starts_with_keyword(title_norm):
                keywords = {"correction", "response", "reply", "appendix", "erratum",
                            "corrigendum", "comment", "addendum", "retraction", "withdrawal"}
                first_word = title_norm.split(' ', 1)[0] if title_norm else ''
                return first_word in keywords

            article_starts_kwd = starts_with_keyword(article_title_norm)
            preprint_starts_kwd = starts_with_keyword(preprint_title_norm)

            if article_starts_kwd != preprint_starts_kwd:
                self.logger.debug(f"Applying title keyword mismatch penalty for input:{input_doi} vs Cand:{preprint_doi} "
                                  f"(inputKwd: {article_starts_kwd}, CandKwd: {preprint_starts_kwd})")
                score *= 0.7

            return score

        except Exception as e:
            self.logger.error(f"Error calculating title score for input:{input_doi} vs Cand:{preprint_doi}: {e}", exc_info=False)
            return None

    def authors_score(self, article_datacite, preprint_crossref):
        input_doi = article_datacite.get(
            "id") or article_datacite.get("doi", "N/A")
        preprint_doi = preprint_crossref.get("DOI", "N/A")

        try:
            attributes = article_datacite.get("attributes", {})
            datacite_people = []
            if isinstance(attributes, dict):
                creators = attributes.get("creators", [])
                contributors = attributes.get("contributors", [])
                if isinstance(creators, list):
                    datacite_people.extend(creators)
                if isinstance(contributors, list):
                    datacite_people.extend([p for p in contributors if isinstance(
                        p, dict) and p.get("nameType") == "Personal"])
            else:
                self.logger.warning(f"Missing or invalid attributes for author extraction in input:{input_doi}")

            crossref_author_list = preprint_crossref.get("author", [])
            if not isinstance(crossref_author_list, list):
                self.logger.warning(f"Crossref 'author' field not a list for Cand:{preprint_doi}. Treating as empty.")
                crossref_author_list = []

            article_authors_norm = self._normalize_authors(
                datacite_people, 'datacite', input_doi)
            preprint_authors_norm = self._normalize_authors(
                crossref_author_list, 'crossref', preprint_doi)

            if article_authors_norm is None or preprint_authors_norm is None:
                self.logger.error(f"Author normalization failed for input:{input_doi} or Cand:{preprint_doi}. Cannot calculate score.")
                return None

            len1 = len(article_authors_norm)
            len2 = len(preprint_authors_norm)

            if len1 == 0 and len2 == 0:
                self.logger.debug(f"Both author lists empty for input:{input_doi} vs Cand:{preprint_doi}. Score: 0.5")
                return 0.5
            if len1 == 0 or len2 == 0:
                penalty_score = 0.1 if max(len1, len2) < 3 else 0.0
                self.logger.debug(f"One author list empty for input:{input_doi} (len={len1}) vs Cand:{preprint_doi} (len={len2}). Score: {penalty_score}")
                return penalty_score

            total_authors = len1 + len2
            author_list_heuristic_threshold = 50
            if total_authors > author_list_heuristic_threshold:
                self.logger.debug(f"Using family name heuristic for large author list ({total_authors}) for input:{input_doi} vs Cand:{preprint_doi}")
                article_families_norm = " ".join(
                    sorted([a.get('family', '') for a in article_authors_norm if a.get('family')]))
                preprint_families_norm = " ".join(
                    sorted([a.get('family', '') for a in preprint_authors_norm if a.get('family')]))

                if not article_families_norm or not preprint_families_norm:
                    self.logger.debug(f"Family name heuristic failed due to missing names for input:{input_doi} vs Cand:{preprint_doi}.")
                    return 0.0
                score = fuzz.token_sort_ratio(
                    article_families_norm, preprint_families_norm) / 100.0
                self.logger.debug(f"Large list heuristic score: {score:.3f}")
                return score

            score_sum = 0.0
            remaining_article_authors = article_authors_norm[:]
            remaining_preprint_authors = preprint_authors_norm[:]
            match_count = 0

            while remaining_article_authors and remaining_preprint_authors:
                best_score, idx1, idx2 = self._find_most_similar_author_pair(
                    remaining_article_authors, remaining_preprint_authors
                )

                similarity_threshold_for_match = 0.5
                if best_score < similarity_threshold_for_match:
                    if best_score >= 0:
                        self.logger.debug(f"Stopping author matching: Best remaining score ({best_score:.3f}) below threshold ({similarity_threshold_for_match}).")
                    break

                score_sum += best_score
                match_count += 1

                if 0 <= idx1 < len(remaining_article_authors) and 0 <= idx2 < len(remaining_preprint_authors):
                    if idx1 >= idx2:
                        del remaining_article_authors[idx1]
                        del remaining_preprint_authors[idx2]
                    else:
                        del remaining_preprint_authors[idx2]
                        del remaining_article_authors[idx1]
                else:
                    self.logger.error(f"Invalid indices ({idx1}, {idx2}) during author matching "
                                       f"(Lists lengths: {len(remaining_article_authors)}, {len(remaining_preprint_authors)}) "
                                       f"for input:{input_doi} vs Cand:{preprint_doi}. Stopping match.")
                    break

            if total_authors > 0:
                final_score = (2.0 * score_sum) / total_authors
            else:
                final_score = 0.5

            final_score = max(0.0, min(1.0, final_score))
            return final_score

        except Exception as e:
            self.logger.error(f"Error calculating authors score for input:{input_doi} vs Cand:{preprint_doi}: {e}", exc_info=False)
            return None

    def _normalize_authors(self, authors_list, source_format, context_doi="N/A"):
        normalized = []
        if not isinstance(authors_list, list):
            self.logger.warning(f"Author list is not a list in {source_format} data for DOI {context_doi}.")
            return []

        try:
            for i, person in enumerate(authors_list):
                if not isinstance(person, dict):
                    self.logger.debug(f"Skipping non-dict author item {i} in {source_format} for DOI {context_doi}")
                    continue

                if source_format == 'datacite' and person.get("nameType") == "Organizational":
                    self.logger.debug(f"Skipping Organizational author/contributor (DataCite) for DOI {context_doi}: {person.get('name')}")
                    continue
                elif source_format == 'crossref' and (person.get('family') is None and person.get('given') is None and person.get('name') is not None):
                    self.logger.debug(f"Skipping potential Organizational author (Crossref heuristic) for DOI {context_doi}: {person.get('name')}")
                    continue

                raw_given = ''
                raw_family = ''
                raw_name_field = ''

                if source_format == 'datacite':
                    raw_given = person.get('givenName', '') or ''
                    raw_family = person.get('familyName', '') or ''
                    raw_name_field = person.get('name', '') or ''

                    if raw_name_field and not raw_family and not raw_given:
                        if ',' in raw_name_field:
                            parts = raw_name_field.split(',', 1)
                            raw_family = parts[0].strip()
                            raw_given = parts[1].strip()
                        else:
                            parts = raw_name_field.split()
                            if len(parts) > 1:
                                raw_family = parts[-1]
                                raw_given = " ".join(parts[:-1])

                elif source_format == 'crossref':
                    raw_given = person.get('given', '') or ''
                    raw_family = person.get('family', '') or ''
                    raw_name_field = person.get('name', '') or ''

                norm_given = self._normalize_string(raw_given)
                norm_family = self._normalize_string(raw_family)
                norm_name_field = self._normalize_string(raw_name_field)

                final_given = norm_given
                final_family = norm_family

                if not final_family and norm_name_field:
                    if ',' in norm_name_field:
                        parts = norm_name_field.split(',', 1)
                        final_family = parts[0].strip()
                        if not final_given:
                            final_given = parts[1].strip()
                    else:
                        parts = norm_name_field.split()
                        if len(parts) > 1:
                            final_family = parts[-1]
                            if not final_given:
                                final_given = " ".join(parts[:-1])
                        elif len(parts) == 1 and not final_given:
                            final_family = parts[0]

                initials = "".join(part[0]
                                   for part in re.findall(r'\b\w', final_given))

                orcid_val = ''
                if source_format == 'datacite':
                    name_identifiers = person.get('nameIdentifiers', [])
                    if isinstance(name_identifiers, list):
                        for identifier in name_identifiers:
                            if isinstance(identifier, dict):
                                scheme = (identifier.get(
                                    'nameIdentifierScheme', '') or '').upper()
                                uri = (identifier.get(
                                    'schemeUri', '') or '').lower()
                                if scheme == 'ORCID' or 'orcid.org' in uri:
                                    orcid_val = identifier.get(
                                        'nameIdentifier', '') or ''
                                    if orcid_val:
                                        break
                elif source_format == 'crossref':
                    orcid_val = person.get('ORCID', '') or ''
                    if orcid_val and 'orcid.org' in orcid_val:
                        orcid_val = orcid_val.split('orcid.org/')[-1]

                norm_orcid = ''
                if orcid_val:
                    cleaned_orcid = re.sub(
                        r'[^0-9X]', '', str(orcid_val).strip().upper())
                    if re.match(r"^\d{15}[\dX]$", cleaned_orcid):
                        norm_orcid = cleaned_orcid

                if final_family or final_given or norm_orcid:
                    author_entry = {
                        'given': final_given,
                        'family': final_family,
                        'initials': initials,
                        'orcid': norm_orcid
                    }
                    name_variations = self._get_author_name_variations(
                        author_entry)
                    if name_variations is None:
                        self.logger.error(f"Failed to generate name variations for author {author_entry} in {source_format} for {context_doi}")
                        continue
                    author_entry['name_variations'] = name_variations
                    normalized.append(author_entry)

            return normalized

        except Exception as e:
            self.logger.error(f"Error during author normalization (format: {source_format}, DOI: {context_doi}): {e}", exc_info=False)
            return []

    def _find_most_similar_author_pair(self, authors1_norm, authors2_norm):
        if not authors1_norm or not authors2_norm:
            return -1.0, -1, -1

        best_score = -1.0
        best_i1 = -1
        best_i2 = -1

        for i1, a1 in enumerate(authors1_norm):
            current_best_score_for_a1 = -1.0
            current_best_i2_for_a1 = -1

            for i2, a2 in enumerate(authors2_norm):
                try:
                    score = self._score_normalized_author_similarity(a1, a2)
                    if score is None:
                        self.logger.debug(f"Author similarity calculation failed between {a1.get('family')} and {a2.get('family')}")
                        continue

                    if score > current_best_score_for_a1:
                        current_best_score_for_a1 = score
                        current_best_i2_for_a1 = i2

                    if current_best_score_for_a1 >= 0.999:
                        break

                except Exception as e:
                    self.logger.error(f"Unexpected error comparing author pair ({a1.get('family')}, {a2.get('family')}): {e}", exc_info=False)
                    continue

            if current_best_score_for_a1 > best_score:
                best_score = current_best_score_for_a1
                best_i1 = i1
                best_i2 = current_best_i2_for_a1

            if best_score >= 0.999:
                break

        final_best_score = max(0.0, best_score) if best_score > -1.0 else -1.0
        return final_best_score, best_i1, best_i2

    def _score_normalized_author_similarity(self, author1_norm, author2_norm):
        try:
            orcid1 = author1_norm.get('orcid')
            orcid2 = author2_norm.get('orcid')
            if orcid1 and orcid2:
                score = 1.0 if orcid1 == orcid2 else 0.0
                return score

            names1 = author1_norm.get('name_variations')
            names2 = author2_norm.get('name_variations')

            if names1 is None or names2 is None:
                self.logger.warning(f"Missing pre-computed name variations for similarity scoring between '{author1_norm.get('family')}' and '{author2_norm.get('family')}'. Cannot score.")
                return None

            if not names1 or not names2:
                f1 = author1_norm.get('family')
                f2 = author2_norm.get('family')
                if f1 and f2 and f1 == f2:
                    self.logger.debug(f"Name variations missing, but families match ('{f1}'). Assigning low score 0.3.")
                    return 0.3
                else:
                    return 0.0

            best_name_score = 0.0
            for n1 in names1:
                current_best_for_n1 = 0.0
                for n2 in names2:
                    score = fuzz.token_sort_ratio(n1, n2) / 100.0
                    if score > current_best_for_n1:
                        current_best_for_n1 = score
                    if current_best_for_n1 > 0.999:
                        break
                if current_best_for_n1 > best_name_score:
                    best_name_score = current_best_for_n1
                if best_name_score > 0.999:
                    break

            f1 = author1_norm.get('family')
            f2 = author2_norm.get('family')
            if f1 and f2 and f1 == f2 and best_name_score > 0.6:
                boosted_score = min(1.0, best_name_score * 1.1)
                best_name_score = boosted_score

            return best_name_score

        except Exception as e:
            self.logger.error(f"Error scoring normalized author similarity between "
                              f"'{author1_norm.get('family')}' and '{author2_norm.get('family')}': {e}", exc_info=False)
            return None

    def _get_author_name_variations(self, author_norm):
        try:
            names = set()
            g = author_norm.get('given', '') or ''
            f = author_norm.get('family', '') or ''
            initials = author_norm.get('initials', '') or ''
            initials_nospace = re.sub(r'[\s.]+', '', initials)

            if f and len(f) > 1:
                names.add(f)

            if g and f:
                full_gf = f"{g} {f}"
                full_fg = f"{f} {g}"
                names.add(full_gf)
                names.add(full_fg)

            if initials_nospace and f:
                initials_space = " ".join(initials_nospace)
                names.add(f"{initials_nospace} {f}")
                names.add(f"{initials_space} {f}")
                names.add(f"{f} {initials_nospace}")
                names.add(f"{f} {initials_space}")

                if initials_nospace:
                    first_initial = initials_nospace[0]
                    names.add(f"{first_initial} {f}")
                    names.add(f"{f} {first_initial}")

            if g and len(g) > 1 and g != f:
                names.add(g)

            if initials_nospace and len(initials_nospace) > 1:
                names.add(initials_nospace)
                names.add(" ".join(initials_nospace))

            final_names = {self._normalize_string(name) for name in names}
            final_names = {name for name in final_names if len(name) > 1}

            if not final_names:
                if f and len(f) == 1 and not g:
                    final_names.add(f)
                if g and len(g) == 1 and not f:
                    final_names.add(g)

            return final_names if final_names else set()

        except Exception as e:
            self.logger.error(f"Error getting author name variations for {str(author_norm)[:50]}: {e}", exc_info=False)
            return None
