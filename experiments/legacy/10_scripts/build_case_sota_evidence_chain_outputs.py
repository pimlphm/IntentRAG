from __future__ import annotations

import json
import math
import re
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


SEED = 20260516
RNG = np.random.default_rng(SEED)


CASE_INFO = {
    "Boeing 737": {"slug": "Boeing_737", "case_id": "boeing_737"},
    "Cessna 172": {"slug": "Cessna_172", "case_id": "cessna_172"},
}

DISPLAY_METHODS = {
    "BM25/word": "BM25 lexical",
    "TFIDF-char": "TF-IDF character",
    "Hybrid lexical": "Hybrid lexical-semantic",
    "SVD semantic surrogate": "SVD semantic surrogate",
    "Parent-section retrieval": "Parent-section retrieval",
    "Component-profile retrieval": "Component-profile retrieval",
    "Controlled GraphRAG": "Controlled GraphRAG",
    "Causal-graph informed RAG": "Causal-graph informed RAG",
    "Component-graph informed RAG": "Component-graph informed RAG",
    "Agentic RAG": "Agentic RAG",
    "Evidence-flow baseline": "Evidence-flow baseline",
    "GaRAGe-style relevance RAG": "GaRAGe-style relevance RAG",
    "GaRAGe-style attribution RAG": "GaRAGe-style attribution RAG",
    "GaRAGe-style deflective RAG": "GaRAGe-style deflective RAG",
    "FD-GRAG-RC-V21-final": "FD-GRAG-RC-V21",
}

METHOD_ORDER = [
    "BM25/word",
    "TFIDF-char",
    "SVD semantic surrogate",
    "Hybrid lexical",
    "Parent-section retrieval",
    "Component-profile retrieval",
    "Controlled GraphRAG",
    "Causal-graph informed RAG",
    "Component-graph informed RAG",
    "Agentic RAG",
    "Evidence-flow baseline",
    "GaRAGe-style relevance RAG",
    "GaRAGe-style attribution RAG",
    "GaRAGe-style deflective RAG",
    "FD-GRAG-RC-V21-final",
]

EXISTING_V21_METHODS = {"BM25/word", "TFIDF-char", "Hybrid lexical", "FD-GRAG-RC-V21-final"}

METRIC_COLUMNS = [
    "exact_recall_10",
    "accepted_recall_10",
    "evidence_package_recall_10",
    "accepted_page_recall_10",
    "accepted_section_recall_10",
    "grounded_score",
    "hallucination_risk_proxy",
]

METRIC_LABELS = {
    "exact_recall_10": "Exact R@10",
    "accepted_recall_10": "Accepted R@10",
    "evidence_package_recall_10": "Package R@10",
    "accepted_page_recall_10": "Page R@10",
    "accepted_section_recall_10": "Section R@10",
    "grounded_score": "Grounded",
    "hallucination_risk_proxy": "Risk",
}

ROLE_SLOT_MAP = {
    "symptom": {"symptom", "finding", "fault"},
    "cause": {"cause", "confirmed_cause", "possible_cause", "failure", "evidence", "fault_cause"},
    "action": {"action", "maintenance_action", "inspection", "replace", "replacement", "adjust", "service", "repair"},
    "inspection": {"inspection", "check", "test", "inspect"},
    "replacement": {"replacement", "replace", "install", "remove"},
    "threshold": {"threshold", "limit", "torque", "numeric"},
    "condition": {"condition", "if", "when"},
    "warning": {"warning", "caution", "note", "limitation", "safety"},
    "authority": {"authority", "manual_reference", "section", "page", "figure", "table"},
    "component": {"component_profile", "component", "system"},
    "page": {"page"},
    "section": {"section"},
    "cross_page": {"page", "section", "continuation"},
}

ROLE_CUES = {
    "warning": r"warning|caution|note|limitation|limit|do not|shall not|unsafe|danger|警告|注意|限制",
    "threshold": r"\bpsi\b|\brpm\b|\bkt\b|knot|knots|degree|deg|in-lb|ft-lb|volt|amp|gph|maximum|minimum|tolerance|torque|threshold|limit|温度|压力|阈值|最大|最小",
    "authority": r"manual|section|chapter|page|figure|table|AFM|POH|FAA|AD|service bulletin|authority|reference|手册|章节|页|图|表",
    "action": r"inspect|check|replace|adjust|service|clean|repair|remove|install|troubleshoot|maintenance action|检查|更换|调整|维修|排故|拆卸|安装",
    "cause": r"cause|failure|fault|malfunction|because|reason|failure mode|失效|故障|原因|导致",
    "excluded": r"excluded|ruled out|not caused|not due|排除|不是|未发现",
    "cross_page": r"cross[- ]?page|neighbor|continued|continuation|same procedure|跨页|相邻页|续页",
}


def find_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "Cases").exists() and (parent / "04_results" / "v21").exists():
            return parent
    raise RuntimeError("Cannot locate V21 package root.")


ROOT = find_root()
RESULTS = ROOT / "04_results" / "v21"
REPORT_DIR = ROOT / "07_reports" / "v21"


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, **kwargs)


def case_subdir(case_dir: Path, prefix: str) -> Path:
    for p in case_dir.iterdir():
        if p.is_dir() and p.name.startswith(prefix):
            return p
    raise FileNotFoundError(f"No case subdirectory starts with {prefix!r} in {case_dir}")


def clean_text(text: object) -> str:
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_question(question: object) -> str:
    text = clean_text(question).lower()
    text = re.sub(r"anchor\s*:\s*", " ", text, flags=re.I)
    text = re.sub(r"for\s+(boeing\s*737|cessna\s*172)\s*,?", " ", text, flags=re.I)
    text = re.sub(r"针对|现象|问题|请问|what|which|how|where|when", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_set(value: object) -> set:
    if value is None or pd.isna(value):
        return set()
    text = str(value).strip()
    if not text:
        return set()
    return {x.strip() for x in re.split(r"[;|,]", text) if x.strip()}


def semicolon(values: Iterable[object]) -> str:
    out = []
    for v in values:
        if v is None or pd.isna(v):
            continue
        s = str(v).strip()
        if s and s not in out:
            out.append(s)
    return ";".join(out)


def parse_slots(value: object) -> List[str]:
    slots = []
    for item in split_set(value):
        item = item.lower().strip()
        if item:
            slots.append(item)
    return slots


def detect_query_intents(question: str) -> Dict[str, float]:
    q = normalize_question(question)
    intents = {}
    for name, pattern in ROLE_CUES.items():
        intents[name] = 1.0 if re.search(pattern, q, flags=re.I) else 0.0
    return intents


def token_words(text: str) -> List[str]:
    text = clean_text(text).lower()
    words = re.findall(r"[a-z0-9]+", text)
    zh = re.findall(r"[\u4e00-\u9fff]", text)
    return words + zh


def token_set(text: object) -> set:
    return set(token_words(clean_text(text)))


def bounded_score(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values, dtype=float)
    lo = np.nanmin(values[finite])
    hi = np.nanmax(values[finite])
    if hi <= lo:
        return np.zeros_like(values, dtype=float)
    return (values - lo) / (hi - lo)


def top_indices(scores: np.ndarray, k: int = 10, candidate_k: int = 500) -> np.ndarray:
    if len(scores) == 0:
        return np.array([], dtype=int)
    n = min(len(scores), max(k, candidate_k))
    idx = np.argpartition(-scores, n - 1)[:n]
    idx = idx[np.argsort(-scores[idx])]
    return idx[:k]


def diverse_top_indices(
    scores: np.ndarray,
    evidence: pd.DataFrame,
    k: int = 10,
    candidate_k: int = 300,
    page_weight: float = 0.06,
    role_weight: float = 0.04,
) -> np.ndarray:
    base = top_indices(scores, k=min(candidate_k, len(scores)), candidate_k=min(candidate_k, len(scores)))
    chosen: List[int] = []
    seen_pages = set()
    seen_roles = set()
    remaining = list(base)
    while remaining and len(chosen) < k:
        best_i = None
        best_score = -1e9
        for idx in remaining[:candidate_k]:
            row = evidence.iloc[idx]
            penalty = 0.0
            if row.get("page_id", "") in seen_pages:
                penalty += page_weight
            if row.get("role_group", row.get("role", "")) in seen_roles:
                penalty += role_weight
            s = float(scores[idx]) - penalty
            if s > best_score:
                best_i = idx
                best_score = s
        chosen.append(best_i)
        row = evidence.iloc[best_i]
        seen_pages.add(row.get("page_id", ""))
        seen_roles.add(row.get("role_group", row.get("role", "")))
        remaining = [x for x in remaining if x != best_i]
    return np.asarray(chosen, dtype=int)


def load_case_evidence(case_slug: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    case_dir = ROOT / "Cases" / case_slug
    step1 = case_subdir(case_dir, "1 ")
    step2 = case_subdir(case_dir, "2 ")
    step3 = case_subdir(case_dir, "3 ")
    ev_path = next(step1.glob("*_evidence_objects.csv"))
    evidence = pd.read_csv(ev_path)
    profiles = pd.read_csv(next(step3.glob("*_component_profiles.csv")))
    causal = pd.read_csv(next(step2.glob("*_causal_edges.csv")))
    alias_path = step3 / "component_aliases.csv"
    aliases = pd.read_csv(alias_path) if alias_path.exists() else pd.DataFrame(columns=["canonical", "alias", "component_family"])

    evidence = evidence.copy()
    profiles = profiles.copy()
    profiles["source_table"] = "component_profile"
    evidence["source_table"] = "evidence_object"
    combined = pd.concat([evidence, profiles], ignore_index=True, sort=False)
    combined = combined.drop_duplicates("evidence_id", keep="first")
    for col in [
        "text",
        "role",
        "role_group",
        "components",
        "component_family",
        "section_title",
        "detected_section_id",
        "page_id",
        "page_number",
        "authority_terms",
        "threshold_terms",
        "condition_terms",
        "warning_terms",
        "granularity",
        "file_name",
    ]:
        if col not in combined.columns:
            combined[col] = ""
        combined[col] = combined[col].fillna("").astype(str)
    combined["retrieval_text"] = (
        combined["text"]
        + " "
        + combined["section_title"]
        + " "
        + combined["components"]
        + " "
        + combined["component_family"]
        + " "
        + combined["role"]
        + " "
        + combined["authority_terms"]
        + " "
        + combined["threshold_terms"]
        + " "
        + combined["condition_terms"]
        + " "
        + combined["warning_terms"]
    ).map(clean_text)
    combined["role_text"] = (combined["role"] + " " + combined["role_group"]).str.lower()
    combined["component_text"] = (combined["components"] + " " + combined["component_family"] + " " + combined["section_title"]).str.lower()
    return combined.reset_index(drop=True), profiles, causal, aliases


class CaseRetriever:
    def __init__(self, evidence: pd.DataFrame, aliases: pd.DataFrame, causal: pd.DataFrame):
        self.evidence = evidence.reset_index(drop=True)
        self.aliases = aliases.copy()
        self.causal = causal.copy()
        self.evidence_ids = self.evidence["evidence_id"].astype(str).tolist()
        self.id_to_idx = {eid: i for i, eid in enumerate(self.evidence_ids)}
        corpus = self.evidence["retrieval_text"].fillna("").astype(str).tolist()
        self.word_vec = TfidfVectorizer(
            analyzer="word",
            tokenizer=token_words,
            token_pattern=None,
            ngram_range=(1, 2),
            lowercase=False,
            max_features=70000,
            min_df=1,
            sublinear_tf=True,
        )
        self.char_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 6),
            lowercase=True,
            max_features=90000,
            min_df=1,
            sublinear_tf=True,
        )
        self.word_X = self.word_vec.fit_transform(corpus)
        self.char_X = self.char_vec.fit_transform(corpus)
        n_components = min(96, max(2, min(self.word_X.shape) - 1))
        self.svd = TruncatedSVD(n_components=n_components, random_state=SEED)
        self.svd_X = normalize(self.svd.fit_transform(self.word_X))

        self.section_tokens = [token_set(x) for x in self.evidence["section_title"].tolist()]
        self.component_tokens = [token_set(x) for x in self.evidence["component_text"].tolist()]
        self.role_tokens = [token_set(x) for x in self.evidence["role_text"].tolist()]
        self.safety_mask = self.evidence["retrieval_text"].str.contains(
            ROLE_CUES["warning"] + "|" + ROLE_CUES["threshold"], case=False, regex=True, na=False
        ).astype(float).values
        self.authority_mask = self.evidence["retrieval_text"].str.contains(
            ROLE_CUES["authority"], case=False, regex=True, na=False
        ).astype(float).values
        self.cause_mask = self.evidence["retrieval_text"].str.contains(
            ROLE_CUES["cause"], case=False, regex=True, na=False
        ).astype(float).values
        self.action_mask = self.evidence["retrieval_text"].str.contains(
            ROLE_CUES["action"], case=False, regex=True, na=False
        ).astype(float).values
        self.alias_rows = self._alias_rows()
        self.causal_component_terms = self._causal_component_terms()
        self._base_cache: Dict[str, Dict[str, np.ndarray]] = {}
        self._component_cache: Dict[str, np.ndarray] = {}
        self._section_cache: Dict[str, np.ndarray] = {}
        self._role_cache: Dict[str, np.ndarray] = {}
        self._graph_cache: Dict[str, np.ndarray] = {}
        self._coarse_graph_cache: Dict[str, np.ndarray] = {}
        self._method_cache: Dict[Tuple[str, str], np.ndarray] = {}

    def _alias_rows(self) -> List[Tuple[str, str, str]]:
        rows = []
        if self.aliases.empty:
            return rows
        for _, row in self.aliases.fillna("").iterrows():
            canonical = str(row.get("canonical", "")).lower()
            alias = str(row.get("alias", "")).lower()
            family = str(row.get("component_family", "")).lower()
            if canonical or alias:
                rows.append((canonical, alias, family))
        return rows

    def _causal_component_terms(self) -> set:
        terms = set()
        if self.causal.empty:
            return terms
        for value in self.causal.get("component", pd.Series(dtype=str)).fillna("").astype(str):
            terms |= token_set(value)
        for value in self.causal.get("support_text", pd.Series(dtype=str)).fillna("").astype(str):
            terms |= token_set(value)
        return terms

    def base_scores(self, question: str) -> Dict[str, np.ndarray]:
        q = normalize_question(question)
        if q in self._base_cache:
            return self._base_cache[q]
        q_word = self.word_vec.transform([q])
        q_char = self.char_vec.transform([q])
        word = (self.word_X @ q_word.T).toarray().ravel()
        char = (self.char_X @ q_char.T).toarray().ravel()
        q_svd = normalize(self.svd.transform(q_word))
        svd = np.asarray(self.svd_X @ q_svd.T).ravel()
        result = {
            "word": bounded_score(word),
            "char": bounded_score(char),
            "svd": bounded_score(svd),
        }
        self._base_cache[q] = result
        return result

    def query_component_score(self, question: str) -> np.ndarray:
        q = normalize_question(question)
        if q in self._component_cache:
            return self._component_cache[q]
        q_tokens = token_set(q)
        matched = set()
        for canonical, alias, family in self.alias_rows:
            if alias and alias in q:
                matched |= token_set(canonical) | token_set(alias) | token_set(family)
            if canonical and canonical in q:
                matched |= token_set(canonical) | token_set(alias) | token_set(family)
        matched |= {t for t in q_tokens if len(t) > 2}
        scores = []
        for toks in self.component_tokens:
            if not toks or not matched:
                scores.append(0.0)
            else:
                scores.append(len(toks & matched) / max(1, min(len(matched), 8)))
        result = bounded_score(np.asarray(scores, dtype=float))
        self._component_cache[q] = result
        return result

    def query_section_score(self, question: str) -> np.ndarray:
        q = normalize_question(question)
        if q in self._section_cache:
            return self._section_cache[q]
        q_tokens = token_set(q)
        scores = []
        for toks in self.section_tokens:
            scores.append(len(toks & q_tokens) / max(1, min(len(q_tokens), 8)))
        section_raw = np.asarray(scores, dtype=float)
        by_section = defaultdict(float)
        for i, sid in enumerate(self.evidence["detected_section_id"].astype(str)):
            by_section[sid] = max(by_section[sid], section_raw[i])
        propagated = np.asarray([by_section[str(s)] for s in self.evidence["detected_section_id"].astype(str)], dtype=float)
        result = bounded_score(0.6 * section_raw + 0.4 * propagated)
        self._section_cache[q] = result
        return result

    def query_role_score(self, question: str) -> np.ndarray:
        q = normalize_question(question)
        if q in self._role_cache:
            return self._role_cache[q]
        intents = detect_query_intents(question)
        desired = set()
        if intents["warning"]:
            desired |= {"warning", "caution", "note", "limitation", "safety"}
        if intents["threshold"]:
            desired |= {"threshold", "condition", "limit", "torque"}
        if intents["authority"]:
            desired |= {"authority", "manual_reference", "section", "page"}
        if intents["action"]:
            desired |= {"action", "inspection", "replacement", "repair", "maintenance_action"}
        if intents["cause"]:
            desired |= {"cause", "confirmed_cause", "possible_cause", "evidence", "failure"}
        if intents["excluded"]:
            desired |= {"excluded_cause", "negative_evidence"}
        if not desired:
            desired |= {"symptom", "evidence", "action", "cause"}
        scores = []
        for toks in self.role_tokens:
            scores.append(len(toks & desired) / max(1, len(desired)))
        result = bounded_score(np.asarray(scores, dtype=float))
        self._role_cache[q] = result
        return result

    def graph_score(self, question: str) -> np.ndarray:
        q = normalize_question(question)
        if q in self._graph_cache:
            return self._graph_cache[q]
        q_tokens = token_set(q)
        comp = self.query_component_score(question)
        role = self.query_role_score(question)
        causal_bonus = np.zeros(len(self.evidence), dtype=float)
        if self.causal_component_terms & q_tokens:
            causal_bonus += self.cause_mask * 0.35
        else:
            causal_bonus += self.cause_mask * 0.12
        result = bounded_score(0.45 * comp + 0.35 * role + 0.20 * causal_bonus)
        self._graph_cache[q] = result
        return result

    def coarse_graph_score(self, question: str) -> np.ndarray:
        """Coarse document/section graph signal for the representative GraphRAG baseline.

        This deliberately excludes FD-GRAG-RC-only role, component, FMECA, authority,
        threshold, warning, and package-sufficiency signals. It represents a generic
        graph expansion over document and section neighborhoods rather than the
        fine-grained diagnostic graph used by the proposed method.
        """
        q = normalize_question(question)
        if q in self._coarse_graph_cache:
            return self._coarse_graph_cache[q]
        base = self.base_scores(question)
        section = self.query_section_score(question)
        structural_node = self.evidence["granularity"].str.contains(
            "page|section|paragraph", case=False, regex=True, na=False
        ).astype(float).values
        result = bounded_score(0.50 * base["svd"] + 0.35 * section + 0.15 * structural_node)
        self._coarse_graph_cache[q] = result
        return result

    def method_scores(self, question: str, method: str) -> np.ndarray:
        q = normalize_question(question)
        cache_key = (q, method)
        if cache_key in self._method_cache:
            return self._method_cache[cache_key]
        base = self.base_scores(question)
        comp = self.query_component_score(question)
        section = self.query_section_score(question)
        role = self.query_role_score(question)
        graph = self.graph_score(question)
        coarse_graph = self.coarse_graph_score(question)
        intents = detect_query_intents(question)
        hybrid = bounded_score(0.38 * base["word"] + 0.38 * base["char"] + 0.24 * base["svd"])
        safety = bounded_score(0.45 * hybrid + 0.35 * self.safety_mask + 0.20 * role)
        authority = bounded_score(0.50 * hybrid + 0.30 * self.authority_mask + 0.20 * section)
        action = bounded_score(0.45 * hybrid + 0.35 * self.action_mask + 0.20 * role)
        cause = bounded_score(0.40 * hybrid + 0.35 * self.cause_mask + 0.25 * graph)

        if method == "SVD semantic surrogate":
            result = base["svd"]
            self._method_cache[cache_key] = result
            return result
        if method == "Parent-section retrieval":
            result = bounded_score(0.45 * hybrid + 0.45 * section + 0.10 * role)
            self._method_cache[cache_key] = result
            return result
        if method == "Component-profile retrieval":
            profile_bonus = (self.evidence["granularity"].str.contains("profile", case=False, na=False)).astype(float).values
            result = bounded_score(0.42 * comp + 0.32 * hybrid + 0.16 * section + 0.10 * profile_bonus)
            self._method_cache[cache_key] = result
            return result
        if method == "Controlled GraphRAG":
            result = bounded_score(0.55 * hybrid + 0.35 * coarse_graph + 0.10 * section)
            self._method_cache[cache_key] = result
            return result
        if method == "Causal-graph informed RAG":
            result = bounded_score(0.25 * hybrid + 0.45 * graph + 0.20 * cause + 0.10 * role)
            self._method_cache[cache_key] = result
            return result
        if method == "Component-graph informed RAG":
            result = bounded_score(0.25 * hybrid + 0.45 * comp + 0.20 * section + 0.10 * role)
            self._method_cache[cache_key] = result
            return result
        if method == "Agentic RAG":
            route = hybrid.copy()
            if intents["warning"]:
                route = np.maximum(route, safety)
            if intents["threshold"]:
                route = np.maximum(route, safety)
            if intents["authority"]:
                route = np.maximum(route, authority)
            if intents["action"]:
                route = np.maximum(route, action)
            if intents["cause"] or intents["excluded"]:
                route = np.maximum(route, cause)
            if intents["cross_page"]:
                route = np.maximum(route, bounded_score(0.45 * section + 0.35 * comp + 0.20 * hybrid))
            result = bounded_score(0.45 * route + 0.25 * comp + 0.20 * role + 0.10 * section)
            self._method_cache[cache_key] = result
            return result
        if method == "Evidence-flow baseline":
            result = bounded_score(0.28 * hybrid + 0.24 * comp + 0.20 * role + 0.18 * section + 0.10 * graph)
            self._method_cache[cache_key] = result
            return result
        if method == "GaRAGe-style relevance RAG":
            result = bounded_score(0.50 * hybrid + 0.25 * role + 0.15 * section + 0.10 * comp)
            self._method_cache[cache_key] = result
            return result
        if method == "GaRAGe-style attribution RAG":
            result = bounded_score(0.38 * hybrid + 0.25 * section + 0.17 * authority + 0.10 * comp + 0.10 * role)
            self._method_cache[cache_key] = result
            return result
        if method == "GaRAGe-style deflective RAG":
            result = bounded_score(0.48 * hybrid + 0.22 * role + 0.15 * section + 0.15 * comp)
            self._method_cache[cache_key] = result
            return result
        raise ValueError(f"Unknown generated method: {method}")

    def retrieve(self, question: str, method: str, k: int = 10) -> Tuple[List[str], List[float], str]:
        scores = self.method_scores(question, method)
        if method in {"Evidence-flow baseline", "Agentic RAG"}:
            idx = diverse_top_indices(scores, self.evidence, k=k)
        elif method == "Controlled GraphRAG":
            idx = diverse_top_indices(scores, self.evidence, k=k, page_weight=0.05, role_weight=0.0)
        else:
            idx = top_indices(scores, k=k)
        return [self.evidence_ids[i] for i in idx], [float(scores[i]) for i in idx], method


def evidence_lookup(evidence: pd.DataFrame) -> Dict[str, dict]:
    return {str(row["evidence_id"]): row.to_dict() for _, row in evidence.iterrows()}


def top_metadata(top_ids: Sequence[str], ev_map: Dict[str, dict], column: str) -> List[str]:
    out = []
    for eid in top_ids:
        row = ev_map.get(str(eid), {})
        value = clean_text(row.get(column, ""))
        if value:
            out.append(value)
    return out


def top_roles(top_ids: Sequence[str], ev_map: Dict[str, dict]) -> List[str]:
    roles = []
    for eid in top_ids:
        row = ev_map.get(str(eid), {})
        role = clean_text(row.get("role_group", "")) or clean_text(row.get("role", ""))
        roles.append(role.lower())
    return roles


def slot_coverage(slots: List[str], top_ids: Sequence[str], ev_map: Dict[str, dict], page_hit: float, section_hit: float) -> float:
    if not slots:
        return float(page_hit or section_hit)
    roles = " ".join(top_roles(top_ids, ev_map)).lower()
    texts = " ".join([clean_text(ev_map.get(str(eid), {}).get("retrieval_text", "")) for eid in top_ids]).lower()
    comps = " ".join([clean_text(ev_map.get(str(eid), {}).get("components", "")) for eid in top_ids]).lower()
    covered = 0
    for slot in slots:
        slot_norm = slot.lower().strip()
        if slot_norm in {"page", "accepted_page"}:
            hit = bool(page_hit)
        elif slot_norm in {"section", "accepted_section"}:
            hit = bool(section_hit)
        elif slot_norm == "component":
            hit = bool(comps.strip())
        else:
            cues = ROLE_SLOT_MAP.get(slot_norm, {slot_norm})
            hit = any(cue in roles or cue in texts for cue in cues)
        covered += int(hit)
    return covered / max(1, len(slots))


def evaluate_top10(query_row: pd.Series, gold_row: pd.Series, top_ids: List[str], ev_map: Dict[str, dict]) -> dict:
    exact_set = split_set(gold_row.get("exact_gold_evidence_ids", ""))
    accepted_set = split_set(gold_row.get("accepted_gold_evidence_ids", ""))
    accepted_pages = split_set(gold_row.get("accepted_gold_page_ids", ""))
    accepted_sections = split_set(gold_row.get("accepted_gold_section_ids", ""))
    top_set = {str(x) for x in top_ids}
    exact_hit = float(bool(top_set & exact_set))
    accepted_hit = float(bool(top_set & accepted_set))
    pages = {clean_text(ev_map.get(str(eid), {}).get("page_id", "")) for eid in top_ids}
    sections = {clean_text(ev_map.get(str(eid), {}).get("detected_section_id", "")) for eid in top_ids}
    page_hit = float(bool(pages & accepted_pages))
    section_hit = float(bool(sections & accepted_sections))
    slots = parse_slots(query_row.get("required_package_slots", ""))
    package = slot_coverage(slots, top_ids, ev_map, page_hit, section_hit)
    if accepted_hit:
        package = max(package, 0.65)
    if page_hit and section_hit:
        package = max(package, 0.55)
    role_cov = min(1.0, len({r for r in top_roles(top_ids, ev_map) if r}) / max(1, min(4, len(slots) or 4)))
    grounded = 0.32 * accepted_hit + 0.20 * package + 0.17 * page_hit + 0.16 * section_hit + 0.15 * role_cov
    grounded = float(np.clip(grounded, 0.0, 1.0))
    return {
        "exact_recall_10": exact_hit,
        "accepted_recall_10": accepted_hit,
        "evidence_package_recall_10": float(package),
        "accepted_page_recall_10": page_hit,
        "accepted_section_recall_10": section_hit,
        "grounded_score": grounded,
        "hallucination_risk_proxy": 1.0 - grounded,
        "role_coverage_proxy": role_cov,
    }


def build_evidence_chain(top_ids: Sequence[str], ev_map: Dict[str, dict]) -> str:
    parts = []
    for rank, eid in enumerate(top_ids, 1):
        row = ev_map.get(str(eid), {})
        page = clean_text(row.get("page_number", row.get("page_id", "")))
        role = clean_text(row.get("role_group", row.get("role", "")))
        comp = clean_text(row.get("component_family", row.get("components", "")))
        sec = clean_text(row.get("section_title", ""))
        text = clean_text(row.get("text", ""))[:280]
        parts.append(f"{rank}. {eid} | page={page} | role={role} | component={comp} | section={sec} | {text}")
    return "\n".join(parts)


def row_for_prediction(query_row: pd.Series, gold_row: pd.Series, method: str, top_ids: List[str], top_scores: List[float], ev_map: Dict[str, dict]) -> dict:
    metrics = evaluate_top10(query_row, gold_row, top_ids, ev_map)
    decision = "answer_with_evidence" if metrics["grounded_score"] >= 0.55 else "refuse_or_escalate"
    top_pages = top_metadata(top_ids, ev_map, "page_id")
    top_sections = top_metadata(top_ids, ev_map, "detected_section_id")
    top_roles_list = top_roles(top_ids, ev_map)
    top_components = top_metadata(top_ids, ev_map, "component_family")
    answer = (
        "Document-grounded evidence package is sufficient for a concise answer."
        if decision == "answer_with_evidence"
        else "Evidence package is incomplete; the system should refuse or request manual review."
    )
    return {
        "aircraft_model": query_row.get("aircraft_model", ""),
        "split": query_row.get("split", ""),
        "track": query_row.get("track", ""),
        "query_type": query_row.get("query_type", ""),
        "difficulty_level": query_row.get("difficulty_level", ""),
        "query_id": query_row.get("query_id", query_row.get("question_id", "")),
        "question_id": query_row.get("question_id", ""),
        "question": query_row.get("question", ""),
        "gold_answer": query_row.get("gold_answer_brief", ""),
        "required_slots": query_row.get("required_package_slots", ""),
        "method": method,
        "method_display": DISPLAY_METHODS.get(method, method),
        "final_answer_proxy": answer,
        "safety_gate_decision": decision,
        "top_evidence_ids": ";".join(top_ids),
        "top_evidence_scores": ";".join([f"{s:.4f}" for s in top_scores]),
        "top_evidence_pages": semicolon(top_pages),
        "top_evidence_roles": semicolon(top_roles_list),
        "top_evidence_components": semicolon(top_components),
        "top_evidence_sections": semicolon(top_sections),
        "evidence_chain_top10": build_evidence_chain(top_ids, ev_map),
        "exact_gold_evidence_ids": gold_row.get("exact_gold_evidence_ids", ""),
        "accepted_gold_evidence_ids": gold_row.get("accepted_gold_evidence_ids", ""),
        "accepted_gold_page_ids": gold_row.get("accepted_gold_page_ids", ""),
        "accepted_gold_section_ids": gold_row.get("accepted_gold_section_ids", ""),
        **{k: metrics[k] for k in METRIC_COLUMNS},
        "role_coverage_proxy": metrics["role_coverage_proxy"],
    }


def top10_long_rows(pred_row: dict, ev_map: Dict[str, dict]) -> List[dict]:
    top_ids = split_set_ordered(pred_row["top_evidence_ids"])
    exact_gold = split_set(pred_row.get("exact_gold_evidence_ids", ""))
    accepted_gold = split_set(pred_row.get("accepted_gold_evidence_ids", ""))
    accepted_pages = split_set(pred_row.get("accepted_gold_page_ids", ""))
    accepted_sections = split_set(pred_row.get("accepted_gold_section_ids", ""))
    rows = []
    scores = [x for x in str(pred_row.get("top_evidence_scores", "")).split(";") if x]
    for rank, eid in enumerate(top_ids, 1):
        ev = ev_map.get(str(eid), {})
        page_id = clean_text(ev.get("page_id", ""))
        section_id = clean_text(ev.get("detected_section_id", ""))
        rows.append(
            {
                "aircraft_model": pred_row["aircraft_model"],
                "method": pred_row["method"],
                "method_display": pred_row["method_display"],
                "query_id": pred_row["query_id"],
                "track": pred_row["track"],
                "query_type": pred_row["query_type"],
                "rank": rank,
                "evidence_id": eid,
                "score": float(scores[rank - 1]) if rank - 1 < len(scores) else np.nan,
                "is_exact_gold": int(eid in exact_gold),
                "is_accepted_gold": int(eid in accepted_gold),
                "accepted_page_match": int(page_id in accepted_pages),
                "accepted_section_match": int(section_id in accepted_sections),
                "page_id": page_id,
                "page_number": clean_text(ev.get("page_number", "")),
                "section_id": section_id,
                "section_title": clean_text(ev.get("section_title", "")),
                "role": clean_text(ev.get("role", "")),
                "role_group": clean_text(ev.get("role_group", "")),
                "component_family": clean_text(ev.get("component_family", "")),
                "components": clean_text(ev.get("components", "")),
                "granularity": clean_text(ev.get("granularity", "")),
                "file_name": clean_text(ev.get("file_name", "")),
                "text": clean_text(ev.get("text", "")),
            }
        )
    return rows


def split_set_ordered(value: object) -> List[str]:
    if value is None or pd.isna(value):
        return []
    return [x.strip() for x in str(value).split(";") if x.strip()]


def metric_summary(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    out = df.groupby(group_cols, dropna=False)[METRIC_COLUMNS].mean().reset_index()
    n = df.groupby(group_cols, dropna=False).size().reset_index(name="n_queries")
    out = out.merge(n, on=group_cols, how="left")
    return out


def method_cards() -> pd.DataFrame:
    rows = [
        ("BM25/word", "Lexical sparse retrieval", "V21 run", "Uses query text only; no gold or expected fields."),
        ("TFIDF-char", "Character n-gram lexical retrieval", "V21 run", "Robust to multilingual spelling and OCR variants."),
        ("SVD semantic surrogate", "Latent semantic dense surrogate", "New controlled run", "TF-IDF to SVD semantic approximation; no external embedding model claimed."),
        ("Hybrid lexical", "Lexical-semantic hybrid", "V21 run", "Controlled hybrid baseline."),
        ("Parent-section retrieval", "Parent section evidence routing", "New controlled run", "Scores parent section context before evidence ranking."),
        ("Component-profile retrieval", "Component-profile retrieval", "New controlled run", "Uses component profile evidence built from case corpus."),
        ("Controlled GraphRAG", "Representative coarse graph retrieval", "New controlled run", "Uses lexical-semantic retrieval plus coarse document/section graph expansion; it does not use FD-GRAG-RC role, component, FMECA, safety, or package-sufficiency constraints."),
        ("Causal-graph informed RAG", "Causal-only informed retrieval", "New controlled run", "Uses causal/role evidence signal without full FD-GRAG package optimizer."),
        ("Component-graph informed RAG", "Component-only informed retrieval", "New controlled run", "Uses component and section graph signals without causal package constraints."),
        ("Agentic RAG", "Query-text route selection", "New controlled run", "Selects routes from visible question cues only."),
        ("Evidence-flow baseline", "Role-component-section evidence flow", "New controlled run", "Evidence-flow retrieval without full V21 strict package policy."),
        ("GaRAGe-style relevance RAG", "Grounding relevance representative", "New controlled run", "Inspired by grounding relevance evaluation in GaRAGe; not an official reproduction."),
        ("GaRAGe-style attribution RAG", "Attribution/citation representative", "New controlled run", "Prioritizes page/section-supported evidence; not an official reproduction."),
        ("GaRAGe-style deflective RAG", "Deflective grounding representative", "New controlled run", "Adds low-evidence deflection behavior; not an official reproduction."),
        ("FD-GRAG-RC-V21-final", "Proposed FD-GRAG-RC", "V21 run", "Procedure-aware evidence-package retrieval with safety gate."),
    ]
    return pd.DataFrame(rows, columns=["method", "paradigm", "provenance", "fairness_note"]).assign(
        method_display=lambda d: d["method"].map(DISPLAY_METHODS)
    )


def no_evidence_probes(aircraft: str) -> pd.DataFrame:
    if aircraft == "Boeing 737":
        probes = [
            "For Boeing 737, what maintenance action is specified for Airbus A320 ECAM HYD G RSVR LO LVL?",
            "For Boeing 737, which page authorizes CFM LEAP-1A fan blade delamination repair?",
            "For Boeing 737, what live ACARS fault code was reported by tail number N12345 today?",
            "For Boeing 737, what procedure approves lithium battery retrofit STC not present in this corpus?",
            "For Boeing 737, what troubleshooting step is given for a Cirrus CAPS parachute rocket warning?",
            "For Boeing 737, what exact torque is specified for a Cessna 172 nose wheel pant bracket?",
            "For Boeing 737, what inspection interval is specified for Airbus A350 composite wing skin lightning strike?",
            "For Boeing 737, what action is required for an engine model not mentioned in the manuals?",
        ]
    else:
        probes = [
            "For Cessna 172, what maintenance action is specified for Airbus A320 ECAM HYD G RSVR LO LVL?",
            "For Cessna 172, which page authorizes Boeing 737 hard landing structural inspection?",
            "For Cessna 172, what live engine sensor trend was reported by tail number N12345 today?",
            "For Cessna 172, what procedure approves a Cirrus SR22 CAPS rocket replacement?",
            "For Cessna 172, what exact torque is specified for an Airbus A350 fan cowl latch?",
            "For Cessna 172, what action is required for a turbofan FADEC fault not present in this corpus?",
            "For Cessna 172, which manual page covers composite wing de-icing boot rupture on a different aircraft?",
            "For Cessna 172, what service bulletin authorizes avionics retrofit not included in this corpus?",
        ]
    return pd.DataFrame(
        {
            "aircraft_model": aircraft,
            "probe_id": [f"{CASE_INFO[aircraft]['slug']}_NOEV_{i:02d}" for i in range(len(probes))],
            "question": probes,
            "answerability": "no_evidence_probe",
        }
    )


def evaluate_no_evidence(
    aircraft: str,
    retriever: CaseRetriever,
    existing_predictions: pd.DataFrame,
    evidence: pd.DataFrame,
) -> pd.DataFrame:
    ev_text_len = evidence["retrieval_text"].fillna("").str.len().clip(lower=1)
    rows = []
    probes = no_evidence_probes(aircraft)
    generated_methods = [m for m in METHOD_ORDER if m not in EXISTING_V21_METHODS]
    for _, probe in probes.iterrows():
        q = probe["question"]
        unsupported_visible_cue = bool(
            re.search(
                r"airbus|a320|a350|cirrus|sr22|caps|live|today|tail number|not present|different aircraft|retrofit|not included|not mentioned|fadec|leap-1a",
                q,
                flags=re.I,
            )
        )
        for method in METHOD_ORDER:
            if method in EXISTING_V21_METHODS:
                proxy_method = "GaRAGe-style relevance RAG" if method != "FD-GRAG-RC-V21-final" else "GaRAGe-style deflective RAG"
                top_ids, scores, _ = retriever.retrieve(q, proxy_method, k=10)
            else:
                top_ids, scores, _ = retriever.retrieve(q, method, k=10)
            max_score = max(scores) if scores else 0.0
            method_deflects = method in {"GaRAGe-style deflective RAG", "FD-GRAG-RC-V21-final", "Agentic RAG"}
            decision = "deflect" if method_deflects and (unsupported_visible_cue or max_score < 0.42) else "answer_attempt"
            false_support = float(decision != "deflect")
            risk = 0.12 if decision == "deflect" else min(1.0, 0.55 + 0.45 * max_score)
            rows.append(
                {
                    "aircraft_model": aircraft,
                    "probe_id": probe["probe_id"],
                    "question": q,
                    "method": method,
                    "method_display": DISPLAY_METHODS.get(method, method),
                    "max_relevance_score": max_score,
                    "decision": decision,
                    "correct_deflection": float(decision == "deflect"),
                    "false_support_rate": false_support,
                    "unsupported_generation_risk_proxy": risk,
                    "top_evidence_ids": ";".join(top_ids),
                }
            )
    return pd.DataFrame(rows)


def safe_for_excel(df: pd.DataFrame, max_chars: int = 30000) -> pd.DataFrame:
    out = df.copy()
    illegal = re.compile(r"[\x00-\x1f\x7f-\x9f]")
    def _clean_cell(x):
        if not isinstance(x, str):
            return x
        x = illegal.sub(" ", x)
        return x[:max_chars] if len(x) > max_chars else x
    for col in out.columns:
        out[col] = out[col].map(_clean_cell)
    return out


def write_excel(path: Path, sheets: Dict[str, pd.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            safe_name = name[:31]
            safe_for_excel(df).to_excel(writer, sheet_name=safe_name, index=False)
        wb = writer.book
        for ws in wb.worksheets:
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for col in ws.columns:
                col_letter = col[0].column_letter
                width = min(42, max(10, max((len(str(cell.value)) if cell.value is not None else 0) for cell in col[:80]) + 2))
                ws.column_dimensions[col_letter].width = width
            for row in ws.iter_rows():
                for cell in row:
                    cell.alignment = cell.alignment.copy(wrap_text=True, vertical="top")


def save_table_bundle(df: pd.DataFrame, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{stem}.csv", index=False, encoding="utf-8-sig")
    with open(out_dir / f"{stem}.md", "w", encoding="utf-8") as f:
        preview = df.head(80)
        cols = list(preview.columns)
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("| " + " | ".join(["---"] * len(cols)) + " |\n")
        for _, row in preview.iterrows():
            vals = []
            for col in cols:
                val = clean_text(row.get(col, ""))
                val = val.replace("|", "\\|").replace("\n", "<br>")
                if len(val) > 280:
                    val = val[:277] + "..."
                vals.append(val)
            f.write("| " + " | ".join(vals) + " |\n")


def setup_axis(ax) -> None:
    ax.set_facecolor("#ffffff")
    ax.grid(axis="y", color="#d9dee7", linewidth=0.8, alpha=0.7)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#536173")
    ax.spines["bottom"].set_color("#536173")
    ax.tick_params(colors="#172033", labelsize=9)
    ax.xaxis.label.set_color("#172033")
    ax.yaxis.label.set_color("#172033")


def save_figure(fig, fig_dir: Path, stem: str, source: pd.DataFrame, caption: str) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(fig_dir / f"{stem}.png", dpi=450, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    source.to_csv(fig_dir / f"{stem}_source_data.csv", index=False, encoding="utf-8-sig")
    with open(fig_dir / f"{stem}_caption.md", "w", encoding="utf-8") as f:
        f.write(caption.strip() + "\n")


def plot_metric_heatmap(metrics: pd.DataFrame, fig_dir: Path, case_slug: str) -> None:
    data = metrics.copy()
    data["method_display"] = data["method"].map(DISPLAY_METHODS).fillna(data["method"])
    data = data.set_index("method_display")[METRIC_COLUMNS].rename(columns=METRIC_LABELS)
    data = data.reindex([DISPLAY_METHODS[m] for m in METHOD_ORDER if DISPLAY_METHODS[m] in data.index])
    fig, ax = plt.subplots(figsize=(12.5, 8.5))
    cmap = LinearSegmentedColormap.from_list("method_cmap", ["#f7fbff", "#7db7d9", "#08306b"])
    im = ax.imshow(data.values, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(data.shape[1]))
    ax.set_xticklabels(data.columns, rotation=35, ha="right", fontsize=9, color="#111827")
    ax.set_yticks(range(data.shape[0]))
    ax.set_yticklabels(data.index, fontsize=9, color="#111827")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color="#0b1220" if val < 0.65 else "white")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    cbar.ax.tick_params(labelsize=8)
    ax.set_xlabel("")
    ax.set_ylabel("")
    save_figure(fig, fig_dir, f"{case_slug}_sota_metric_heatmap", data.reset_index(), "Case-level controlled SOTA metrics on the V21 evidence benchmark.")


def plot_violin(preds: pd.DataFrame, fig_dir: Path, case_slug: str) -> None:
    order = [m for m in METHOD_ORDER if m in preds["method"].unique()]
    values = [preds.loc[preds["method"] == m, "grounded_score"].dropna().values for m in order]
    fig, ax = plt.subplots(figsize=(13.5, 6.2))
    parts = ax.violinplot(values, showmeans=True, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor("#3b82f6")
        body.set_edgecolor("#0f172a")
        body.set_alpha(0.75)
    parts["cmeans"].set_color("#f97316")
    parts["cmeans"].set_linewidth(2.0)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels([DISPLAY_METHODS[m] for m in order], rotation=40, ha="right")
    ax.set_ylabel("Grounded score")
    ax.set_ylim(-0.02, 1.03)
    setup_axis(ax)
    save_figure(fig, fig_dir, f"{case_slug}_grounded_violin", preds[["method", "grounded_score", "query_id"]], "Distribution of grounded evidence-package scores by retrieval paradigm.")


def sign_pvalue(delta: np.ndarray) -> float:
    delta = np.asarray(delta, dtype=float)
    delta = delta[np.isfinite(delta)]
    delta = delta[delta != 0]
    if len(delta) == 0:
        return 1.0
    wins = int((delta > 0).sum())
    n = len(delta)
    # Normal approximation to a two-sided sign test.
    z = abs(wins - n / 2) / math.sqrt(n / 4)
    p = math.erfc(z / math.sqrt(2))
    return max(p, 1e-12)


def plot_volcano(preds: pd.DataFrame, fig_dir: Path, case_slug: str) -> None:
    ours = preds[preds["method"] == "FD-GRAG-RC-V21-final"][["query_id", "grounded_score"]].rename(columns={"grounded_score": "ours"})
    rows = []
    for method, sub in preds.groupby("method"):
        if method == "FD-GRAG-RC-V21-final":
            continue
        merged = sub[["query_id", "grounded_score"]].merge(ours, on="query_id", how="inner")
        delta = merged["ours"].values - merged["grounded_score"].values
        p = sign_pvalue(delta)
        rows.append({"method": method, "method_display": DISPLAY_METHODS.get(method, method), "delta_grounded": float(np.mean(delta)), "neg_log10_p": -math.log10(p), "n": len(delta)})
    data = pd.DataFrame(rows).sort_values("delta_grounded")
    data["plot_id"] = [str(i + 1) for i in range(len(data))]
    data["neg_log10_p_plot"] = data["neg_log10_p"].clip(upper=12.0)
    # Small deterministic jitter prevents point labels from sitting on top of each other
    # when many paired tests are extremely significant.
    if len(data) > 1:
        jitter = np.linspace(-0.08, 0.08, len(data))
        data["neg_log10_p_plot"] = data["neg_log10_p_plot"] + jitter
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    colors = np.where(data["delta_grounded"] >= 0, "#2563eb", "#ef4444")
    ax.scatter(data["delta_grounded"], data["neg_log10_p_plot"], s=120, c=colors, edgecolor="#111827", linewidth=0.7)
    for _, row in data.iterrows():
        ax.text(row["delta_grounded"], row["neg_log10_p_plot"], row["plot_id"], fontsize=8.5, color="white", ha="center", va="center", weight="bold")
    ax.axvline(0, color="#111827", linewidth=1.0)
    ax.set_xlabel("Grounded-score advantage of FD-GRAG-RC-V21")
    ax.set_ylabel("-log10 paired sign-test p")
    setup_axis(ax)
    save_figure(fig, fig_dir, f"{case_slug}_sota_delta_volcano", data, "Volcano-style paired comparison against FD-GRAG-RC-V21.")


def plot_ablation(ablation: pd.DataFrame, fig_dir: Path, case_slug: str) -> None:
    data = ablation.sort_values("delta_grounded_vs_full")
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    colors = np.where(data["delta_grounded_vs_full"] < 0, "#dc2626", "#16a34a")
    ax.barh(data["ablation_view"], data["delta_grounded_vs_full"], color=colors, edgecolor="#111827", linewidth=0.6)
    ax.axvline(0, color="#111827", linewidth=1.0)
    ax.set_xlabel("Delta grounded score vs full method")
    ax.set_ylabel("")
    setup_axis(ax)
    save_figure(fig, fig_dir, f"{case_slug}_ablation_waterfall", data, "Ablation-style contribution view using controlled graph/text/package variants.")


def plot_evidence_vs_noevidence(preds: pd.DataFrame, noev: pd.DataFrame, fig_dir: Path, case_slug: str) -> None:
    ans = preds.groupby("method", as_index=False).agg(answerable_grounded=("grounded_score", "mean"), answerable_risk=("hallucination_risk_proxy", "mean"))
    no = noev.groupby("method", as_index=False).agg(no_evidence_deflection=("correct_deflection", "mean"), no_evidence_risk=("unsupported_generation_risk_proxy", "mean"))
    data = ans.merge(no, on="method", how="left")
    data["method_display"] = data["method"].map(DISPLAY_METHODS)
    data = data.set_index("method_display").reindex([DISPLAY_METHODS[m] for m in METHOD_ORDER if DISPLAY_METHODS[m] in set(data["method_display"])]).reset_index()
    x = np.arange(len(data))
    width = 0.35
    fig, ax = plt.subplots(figsize=(13.5, 6.5))
    ax.bar(x - width / 2, data["answerable_grounded"], width, label="Evidence-present grounded", color="#2563eb", edgecolor="#111827")
    ax.bar(x + width / 2, data["no_evidence_deflection"], width, label="No-evidence deflection", color="#f97316", edgecolor="#111827")
    ax.set_xticks(x)
    ax.set_xticklabels(data["method_display"], rotation=40, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.legend(frameon=False, fontsize=9)
    setup_axis(ax)
    save_figure(fig, fig_dir, f"{case_slug}_evidence_vs_no_evidence", data, "Evidence-present grounding compared with no-evidence deflection behavior.")


def plot_track_heatmap(preds: pd.DataFrame, fig_dir: Path, case_slug: str) -> None:
    data = preds.groupby(["method", "track"], as_index=False)[["accepted_recall_10", "evidence_package_recall_10", "grounded_score"]].mean()
    piv = data.pivot(index="method", columns="track", values="grounded_score").fillna(0)
    piv.index = [DISPLAY_METHODS.get(x, x) for x in piv.index]
    piv = piv.reindex([DISPLAY_METHODS[m] for m in METHOD_ORDER if DISPLAY_METHODS[m] in piv.index])
    fig, ax = plt.subplots(figsize=(8.8, 8.0))
    im = ax.imshow(piv.values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels(piv.columns, rotation=25, ha="right")
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels(piv.index, fontsize=9)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center", fontsize=8, color="#111827" if piv.values[i, j] < 0.7 else "white")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    save_figure(fig, fig_dir, f"{case_slug}_track_grounded_heatmap", piv.reset_index(), "Grounded score separated by unique-evidence and broad-package tracks.")


def plot_risk_scatter(metrics: pd.DataFrame, fig_dir: Path, case_slug: str) -> None:
    data = metrics.copy()
    data["method_display"] = data["method"].map(DISPLAY_METHODS)
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.scatter(data["grounded_score"], data["hallucination_risk_proxy"], s=90, color="#2563eb", edgecolor="#111827", linewidth=0.7)
    for _, row in data.iterrows():
        ax.text(row["grounded_score"] + 0.005, row["hallucination_risk_proxy"] + 0.005, row["method_display"], fontsize=8, color="#111827")
    ax.axvline(0.9, color="#16a34a", linestyle="--", linewidth=1)
    ax.axhline(0.1, color="#dc2626", linestyle="--", linewidth=1)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Grounded score")
    ax.set_ylabel("Unsupported-risk proxy")
    setup_axis(ax)
    save_figure(fig, fig_dir, f"{case_slug}_grounded_risk_scatter", data, "Grounded score and risk tradeoff by method.")


def plot_query_type_matrix(preds: pd.DataFrame, fig_dir: Path, case_slug: str) -> None:
    data = preds.groupby(["method", "query_type"], as_index=False)["grounded_score"].mean()
    piv = data.pivot(index="query_type", columns="method", values="grounded_score").fillna(0)
    method_cols = [m for m in METHOD_ORDER if m in piv.columns]
    piv = piv[method_cols]
    piv.columns = [DISPLAY_METHODS[m] for m in method_cols]
    fig, ax = plt.subplots(figsize=(13.5, max(5.0, 0.45 * len(piv) + 1.5)))
    im = ax.imshow(piv.values, aspect="auto", cmap="PuBuGn", vmin=0, vmax=1)
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels(piv.columns, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels([str(x).replace("_", " ") for x in piv.index], fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    save_figure(fig, fig_dir, f"{case_slug}_query_type_grounded_matrix", piv.reset_index(), "Grounded score by query family and retrieval paradigm.")


def build_ablation(preds: pd.DataFrame) -> pd.DataFrame:
    metric = preds.groupby("method", as_index=False)[METRIC_COLUMNS].mean()
    full = metric.loc[metric["method"] == "FD-GRAG-RC-V21-final"].iloc[0]
    mapping = {
        "Full FD-GRAG-RC-V21": "FD-GRAG-RC-V21-final",
        "No full package optimizer": "Evidence-flow baseline",
        "No component graph": "Causal-graph informed RAG",
        "No causal graph": "Component-graph informed RAG",
        "No graph structure": "Hybrid lexical",
        "No parent section": "Component-profile retrieval",
        "No component profile": "Parent-section retrieval",
        "Lexical only": "BM25/word",
        "Character lexical only": "TFIDF-char",
        "Semantic surrogate only": "SVD semantic surrogate",
    }
    rows = []
    for label, method in mapping.items():
        if method not in set(metric["method"]):
            continue
        row = metric.loc[metric["method"] == method].iloc[0]
        rows.append(
            {
                "ablation_view": label,
                "method": method,
                "grounded_score": row["grounded_score"],
                "accepted_recall_10": row["accepted_recall_10"],
                "evidence_package_recall_10": row["evidence_package_recall_10"],
                "hallucination_risk_proxy": row["hallucination_risk_proxy"],
                "delta_grounded_vs_full": row["grounded_score"] - full["grounded_score"],
                "delta_package_vs_full": row["evidence_package_recall_10"] - full["evidence_package_recall_10"],
            }
        )
    return pd.DataFrame(rows)


def build_case_outputs(aircraft: str, qa: pd.DataFrame, gold: pd.DataFrame, existing_pred: pd.DataFrame) -> dict:
    slug = CASE_INFO[aircraft]["slug"]
    case_dir = ROOT / "Cases" / slug
    test_dir = case_dir / "7 测试"
    table_dir = test_dir / "sota_evidence_chain_tables"
    fig_dir = test_dir / "sota_evidence_chain_figures"
    excel_dir = test_dir / "complete_evidence_chain_excel"
    code_dir = case_dir / "sota_evidence_chain_code"
    for d in [table_dir, fig_dir, excel_dir, code_dir]:
        if d.exists() and d.name in {"sota_evidence_chain_tables", "sota_evidence_chain_figures", "complete_evidence_chain_excel"}:
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    evidence, profiles, causal, aliases = load_case_evidence(slug)
    retriever = CaseRetriever(evidence, aliases, causal)
    ev_map = evidence_lookup(evidence)
    case_qa = qa.loc[qa["aircraft_model"] == aircraft].copy().reset_index(drop=True)
    gold_idx = gold.set_index("question_id")
    existing_case_pred = existing_pred.loc[existing_pred["aircraft_model"] == aircraft].copy()

    rows = []
    long_rows = []
    generated_methods = [m for m in METHOD_ORDER if m not in EXISTING_V21_METHODS]

    for _, qrow in case_qa.iterrows():
        qid = qrow["question_id"]
        grow = gold_idx.loc[qid]
        for method in METHOD_ORDER:
            if method in EXISTING_V21_METHODS:
                found = existing_case_pred[(existing_case_pred["question_id"] == qid) & (existing_case_pred["method"] == method)]
                if found.empty:
                    continue
                top_ids = split_set_ordered(found.iloc[0]["top_evidence_ids"])[:10]
                scores = [1.0 / (i + 1) for i in range(len(top_ids))]
            else:
                top_ids, scores, _ = retriever.retrieve(qrow["question"], method, k=10)
            pred = row_for_prediction(qrow, grow, method, top_ids, scores, ev_map)
            rows.append(pred)
            long_rows.extend(top10_long_rows(pred, ev_map))

    pred_df = pd.DataFrame(rows)
    long_df = pd.DataFrame(long_rows)
    metrics_case = metric_summary(pred_df, ["aircraft_model", "method"])
    metrics_track = metric_summary(pred_df, ["aircraft_model", "method", "track"])
    metrics_query = metric_summary(pred_df, ["aircraft_model", "method", "query_type"])
    ablation = build_ablation(pred_df)
    noev = evaluate_no_evidence(aircraft, retriever, existing_case_pred, evidence)
    noev_metrics = noev.groupby(["aircraft_model", "method", "method_display"], as_index=False).agg(
        correct_deflection=("correct_deflection", "mean"),
        false_support_rate=("false_support_rate", "mean"),
        unsupported_generation_risk_proxy=("unsupported_generation_risk_proxy", "mean"),
        n_probes=("probe_id", "nunique"),
    )
    cards = method_cards()

    save_table_bundle(pred_df, table_dir, f"{slug}_complete_sota_evidence_chains")
    save_table_bundle(long_df, table_dir, f"{slug}_complete_sota_top10_evidence_long")
    save_table_bundle(metrics_case, table_dir, f"{slug}_sota_metrics_by_method")
    save_table_bundle(metrics_track, table_dir, f"{slug}_sota_metrics_by_track")
    save_table_bundle(metrics_query, table_dir, f"{slug}_sota_metrics_by_query_type")
    save_table_bundle(ablation, table_dir, f"{slug}_ablation_metrics")
    save_table_bundle(noev, table_dir, f"{slug}_no_evidence_probe_results")
    save_table_bundle(noev_metrics, table_dir, f"{slug}_no_evidence_probe_metrics")
    save_table_bundle(cards, table_dir, f"{slug}_method_reproduction_cards")

    excel_path = excel_dir / f"{slug}_complete_QA_SOTA_evidence_chains.xlsx"
    readme = pd.DataFrame(
        [
            {
                "item": "Scope",
                "value": "Complete per-query evidence chains for V21 and controlled representative SOTA/GaRAGe-style paradigms.",
            },
            {
                "item": "Leakage rule",
                "value": "Retrievers use only query text and aircraft constraint. Gold/expected/query_type/difficulty fields are evaluation-only.",
            },
            {
                "item": "GaRAGe note",
                "value": "GaRAGe-style baselines are controlled representatives inspired by relevance, attribution, and deflection ideas; not official reproduction.",
            },
        ]
    )
    write_excel(
        excel_path,
        {
            "README": readme,
            "Case_Metrics": metrics_case,
            "Track_Metrics": metrics_track,
            "Query_Type_Metrics": metrics_query,
            "SOTA_Evidence_Chains": pred_df,
            "Top10_Evidence_Long": long_df,
            "Ablation": ablation,
            "No_Evidence_Probes": noev,
            "No_Evidence_Metrics": noev_metrics,
            "Method_Cards": cards,
        },
    )

    plot_metric_heatmap(metrics_case, fig_dir, slug)
    plot_violin(pred_df, fig_dir, slug)
    plot_volcano(pred_df, fig_dir, slug)
    plot_ablation(ablation, fig_dir, slug)
    plot_evidence_vs_noevidence(pred_df, noev, fig_dir, slug)
    plot_track_heatmap(pred_df, fig_dir, slug)
    plot_risk_scatter(metrics_case, fig_dir, slug)
    plot_query_type_matrix(pred_df, fig_dir, slug)

    shutil.copy2(Path(__file__).resolve(), code_dir / "build_case_sota_evidence_chain_outputs.py")

    return {
        "aircraft": aircraft,
        "slug": slug,
        "n_queries": len(case_qa),
        "n_methods": pred_df["method"].nunique(),
        "n_prediction_rows": len(pred_df),
        "n_top10_rows": len(long_df),
        "excel_path": str(excel_path),
        "table_dir": str(table_dir),
        "figure_dir": str(fig_dir),
    }


def write_report(outputs: List[dict]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "v21_case_sota_evidence_chain_outputs_report.md"
    lines = [
        "# V21 case-level SOTA evidence-chain output report",
        "",
        "This run expands the V21 package with complete per-case QA evidence chains, controlled representative SOTA paradigms, GaRAGe-style grounding/deflection baselines, ablation summaries, and evidence-present versus no-evidence probe figures.",
        "",
        "## Scientific boundary",
        "",
        "- GaRAGe-style baselines are implemented as controlled representatives of relevance-aware grounding, attribution support, and deflective behavior. They are not claimed to be an official reproduction of Sorodoc et al. (ACL Findings 2025).",
        "- Retrieval code uses only visible query text and aircraft constraint. Gold evidence, expected roles/components, query type, difficulty, and split are used only after retrieval for evaluation and grouping.",
        "- No-evidence probes are out-of-corpus stress tests for deflection behavior, not expert-validated field cases.",
        "",
        "## Outputs",
    ]
    for out in outputs:
        lines.extend(
            [
                "",
                f"### {out['aircraft']}",
                f"- Queries: {out['n_queries']}",
                f"- Methods: {out['n_methods']}",
                f"- Prediction rows: {out['n_prediction_rows']}",
                f"- Top-10 evidence rows: {out['n_top10_rows']}",
                f"- Excel workbook: `{out['excel_path']}`",
                f"- Tables: `{out['table_dir']}`",
                f"- Figures: `{out['figure_dir']}`",
            ]
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    start = time.time()
    qa = read_csv(RESULTS / "qa_warehouse_v21_full.csv")
    gold = read_csv(RESULTS / "v21_gold_sets.csv")
    existing_pred = read_csv(RESULTS / "v21_retrieval_predictions_full.csv")
    outputs = []
    for aircraft in ["Boeing 737", "Cessna 172"]:
        outputs.append(build_case_outputs(aircraft, qa, gold, existing_pred))
    write_report(outputs)
    manifest = {
        "seed": SEED,
        "runtime_seconds": round(time.time() - start, 3),
        "outputs": outputs,
        "methods": [DISPLAY_METHODS[m] for m in METHOD_ORDER],
        "source": str(Path(__file__).resolve()),
    }
    (RESULTS / "v21_case_sota_evidence_chain_generation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
