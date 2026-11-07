"""Gold-free adapter for the manuscript's 15 controlled retrieval settings.

The 11 structural settings reuse the original CaseRetriever without changing its
score functions. Word/character/hybrid settings reuse V21's original functions.
The V21 final setting is explicitly a *visible-router adaptation*: the old run
selected quotas from a locked ``track`` column; this adapter selects them from
the question's visible cue marker. It must not be reported as exact replication
of that historical track-conditioned result.

None of these settings is an official GraphRAG, RAPTOR, MemGAS, Self-RAG or GaRAGe
reproduction. The old ``BM25/word`` setting is word TF-IDF, not Okapi BM25.

Public input: question, aircraft and the document-derived index. Evaluation
labels are neither accepted by retrieve() nor loaded by build_index().
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd


METHODS = (
    "BM25/word", "TFIDF-char", "SVD semantic surrogate", "Hybrid lexical",
    "Parent-section retrieval", "Component-profile retrieval",
    "Controlled GraphRAG", "Causal-graph informed RAG",
    "Component-graph informed RAG", "Agentic RAG", "Evidence-flow baseline",
    "GaRAGe-style relevance RAG", "GaRAGe-style attribution RAG",
    "GaRAGe-style deflective RAG", "FD-GRAG-RC-V21-final",
)
V21_METHODS = {"BM25/word", "TFIDF-char", "Hybrid lexical", "FD-GRAG-RC-V21-final"}
ONLINE_COLUMNS = (
    "evidence_id", "aircraft_model", "text", "role", "role_group",
    "components", "component_family", "section_title", "detected_section_id",
    "page_id", "page_number", "authority_terms", "threshold_terms",
    "condition_terms", "warning_terms", "granularity", "file_name", "source_id",
    "doc_id", "document_id", "revision", "revision_id", "source_span_hash",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_v21_functions(path: Path) -> ModuleType:
    """Execute selected original definitions, avoiding old output mkdir loops."""
    wanted_functions = {
        "norm", "norm_low", "tokenize", "component_from_question",
        "strip_boilerplate", "is_identifiable_anchor", "cue_segments",
        "build_indexes", "contains_any", "query_scores", "top_indices",
        "route_union_indices",
    }
    wanted_constants = {
        "ACTION_TERMS", "CAUSE_TERMS", "SAFETY_TERMS", "AUTHORITY_TERMS",
        "UNIT_TERMS", "CONDITION_TERMS", "COMPONENT_TERMS", "BOILERPLATE_PATTERNS",
    }
    parsed = ast.parse(path.read_text(encoding="utf-8-sig"), str(path))
    selected = []
    for node in parsed.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            selected.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted_functions:
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in wanted_constants
            for target in node.targets
        ):
            selected.append(node)
    module = ModuleType("legacy_v21_retrieval_definitions")
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), module.__dict__)
    return module


def _prepare_online_evidence(evidence: pd.DataFrame) -> pd.DataFrame:
    missing = {"evidence_id", "aircraft_model", "text"} - set(evidence.columns)
    if missing:
        raise ValueError(f"Document index missing columns: {sorted(missing)}")
    # Strict projection: no qrels/track/query-type/required-slot/split columns.
    ev = evidence.loc[:, [c for c in ONLINE_COLUMNS if c in evidence]].copy()
    ev = ev.drop_duplicates("evidence_id", keep="first").reset_index(drop=True)
    for column in ONLINE_COLUMNS:
        if column not in ev:
            ev[column] = ""
        ev[column] = ev[column].fillna("").astype(str)
    ev["retrieval_text"] = ev[[
        "text", "section_title", "components", "component_family", "role",
        "authority_terms", "threshold_terms", "condition_terms", "warning_terms",
    ]].agg(" ".join, axis=1).str.replace(r"\s+", " ", regex=True).str.strip()
    ev["role_text"] = (ev["role"] + " " + ev["role_group"]).str.lower()
    ev["component_text"] = (
        ev["components"] + " " + ev["component_family"] + " " + ev["section_title"]
    ).str.lower()
    return ev


def method_registry() -> list[dict]:
    registry = []
    for method in METHODS:
        label = method
        note = "Original project controlled implementation; not official reproduction."
        if method == "BM25/word":
            label = "Word TF-IDF (legacy BM25/word)"
            note = "Original V21 word 1-2 gram TF-IDF cosine; does not implement BM25."
        elif method == "FD-GRAG-RC-V21-final":
            label = "IntentRAG V21 visible-router"
            note = "Original score/union functions; quotas inferred from visible cue marker, not locked track."
        registry.append({"method": method, "display_name": label,
                         "implementation_kind": "controlled_project_baseline",
                         "official_reproduction": False, "note": note})
    return registry


def _fast_diverse_top_indices(scores, evidence, k=10, candidate_k=300,
                              page_weight=0.06, role_weight=0.04):
    """Legacy greedy diversity algorithm, with iloc replaced by static lists.

    Candidate order, strict tie comparison and penalty arithmetic are unchanged.
    The original iloc fetched a complete Arrow-backed dataframe row thousands
    of times, although the selector reads only page_id and role_group.
    """
    # Match the original top_indices partial-selection order exactly.
    n = min(candidate_k, len(scores))
    if n == 0:
        return np.array([], dtype=int)
    base = np.argpartition(-scores, n-1)[:n]
    base = base[np.argsort(-scores[base])][:min(candidate_k, len(scores))]
    pages = evidence['page_id'].tolist() if 'page_id' in evidence else ['']*len(evidence)
    role_column = 'role_group' if 'role_group' in evidence else 'role'
    roles = evidence[role_column].tolist() if role_column in evidence else ['']*len(evidence)
    chosen, seen_pages, seen_roles, remaining = [], set(), set(), list(base)
    while remaining and len(chosen) < k:
        best_i, best_score = None, -1e9
        for idx in remaining[:candidate_k]:
            penalty = 0.0
            if pages[idx] in seen_pages:
                penalty += page_weight
            if roles[idx] in seen_roles:
                penalty += role_weight
            score = float(scores[idx])-penalty
            if score > best_score:
                best_i, best_score = idx, score
        chosen.append(best_i)
        seen_pages.add(pages[best_i])
        seen_roles.add(roles[best_i])
        remaining = [x for x in remaining if x != best_i]
    return np.asarray(chosen, dtype=int)


class BaselineIndex:
    def __init__(self, project_root: Path, evidence: pd.DataFrame,
                 aliases: pd.DataFrame, causal: pd.DataFrame):
        self.project_root = project_root
        case_path = project_root / "10_scripts" / "build_case_sota_evidence_chain_outputs.py"
        v21_path = project_root / "_pipeline_v21" / "p0_build_eval_v21.py"
        self.case_module = _load_module(case_path, "legacy_case_retrieval_adapter")
        self.original_diverse_top_indices = self.case_module.diverse_top_indices
        self.case_module.diverse_top_indices = _fast_diverse_top_indices
        self.v21_module = _load_v21_functions(v21_path)
        self.source_hashes = {str(case_path): _sha(case_path), str(v21_path): _sha(v21_path)}
        self.evidence = _prepare_online_evidence(evidence)
        self._case_indexes = {}
        self._v21_indexes = {}
        self._aliases = aliases
        self._causal = causal
        self._query_cache = {}
        self._last_question = None
        self._static_v21 = None

    def _optimized_v21_scores(self, question: str, aircraft: str, ev, features):
        """Strict-equivalent V21 arithmetic, with document-only masks precomputed.

        Joint-corpus fitted IDF is preserved. Multiplication and phrase matching
        need run only for the visible aircraft: every other result is replaced
        by -1e9 in the original function. No retrieval weight changes.
        """
        original = self.v21_module
        texts, word, xw, char, xc = features
        if self._static_v21 is None:
            low = [str(text).lower() for text in ev['text']]
            self._static_v21 = {
                'text_low': low,
                'norm_text_low': [text.lower() for text in texts],
                'role_action': np.asarray([float(original.contains_any(text, original.ACTION_TERMS)) for text in low]),
                'role_cause': np.asarray([float(original.contains_any(text, original.CAUSE_TERMS)) for text in low]),
                'role_safety': np.asarray([float(original.contains_any(text, original.SAFETY_TERMS)) for text in low]),
                'role_auth': np.asarray([float(original.contains_any(text, original.AUTHORITY_TERMS)) for text in low]),
                'role_thr': np.asarray([float(original.contains_any(text, original.UNIT_TERMS)) for text in low]),
                'facet': ev['granularity'].eq('facetpoint').to_numpy(dtype=float),
                'paragraph': ev['granularity'].isin(['paragraph', 'table_row']).to_numpy(dtype=float),
                'by_aircraft': {
                    plane: np.flatnonzero(ev['aircraft_model'].eq(plane).to_numpy())
                    for plane in ev['aircraft_model'].unique()
                },
            }
        static = self._static_v21
        active = static['by_aircraft'][aircraft]
        q_search = original.strip_boilerplate(question)
        qw, qc = word.transform([q_search]), char.transform([q_search])
        word_s, char_s = np.zeros(len(ev)), np.zeros(len(ev))
        word_s[active] = (xw[active] @ qw.T).toarray().ravel()
        char_s[active] = (xc[active] @ qc.T).toarray().ravel()
        cue_s = np.zeros(len(ev), dtype=float)
        for cue in original.cue_segments(question):
            qcw, qcc = word.transform([cue]), char.transform([cue])
            route = 0.42*(xw[active] @ qcw.T).toarray().ravel() + 0.58*(xc[active] @ qcc.T).toarray().ravel()
            cue_s[active] = np.maximum(cue_s[active], route)
        phrase = np.zeros(len(ev))
        toks = [t for t in original.tokenize(q_search) if len(t) > 3]
        phrases = [' '.join(toks[i:i+n]) for n in (8, 6, 4, 3) for i in range(max(0, len(toks)-n+1))]
        phrases = [p for p in phrases if len(p) > 10][:20]
        for i in active:
            phrase[i] = sum(1 for p in phrases if p in static['norm_text_low'][i])/max(1, len(phrases))
        comp = original.component_from_question(question)
        comp_s = np.zeros(len(ev), dtype=float)
        if comp:
            comp_s[active] = [float(comp in static['text_low'][i]) for i in active]
        facet, paragraph = static['facet'], static['paragraph']
        action, cause = static['role_action'], static['role_cause']
        safety, auth, threshold = static['role_safety'], static['role_auth'], static['role_thr']
        scores = {
            'BM25/word': word_s,
            'TFIDF-char': char_s,
            'Hybrid lexical': 0.45*word_s + 0.45*char_s + 0.10*phrase,
            'FD-GRAG-RC-V21-exact': 0.05*word_s + 0.14*char_s + 0.12*phrase + 0.76*cue_s + 0.04*facet + 0.04*paragraph,
            'FD-GRAG-RC-V21-package': 0.14*word_s + 0.18*char_s + 0.30*cue_s + 0.18*phrase + 0.10*comp_s + 0.08*action + 0.08*cause + 0.06*safety + 0.04*auth + 0.04*threshold,
            'FD-GRAG-RC-V21-cue-route': 0.70*cue_s + 0.15*phrase + 0.08*char_s + 0.07*word_s + 0.04*facet,
        }
        mask = ev['aircraft_model'].eq(aircraft).to_numpy()
        for name, score in scores.items():
            arr = score.copy()
            arr[~mask] = -1e9
            scores[name] = arr
        return scores

    def _case(self, aircraft: str):
        if aircraft not in self._case_indexes:
            ev = self.evidence.loc[self.evidence["aircraft_model"].eq(aircraft)].reset_index(drop=True)
            if ev.empty:
                raise ValueError(f"Unknown aircraft: {aircraft}")
            aliases = self._aliases
            if "aircraft_model" in aliases:
                aliases = aliases.loc[aliases["aircraft_model"].eq(aircraft)]
            causal = self._causal
            if "aircraft_model" in causal:
                causal = causal.loc[causal["aircraft_model"].eq(aircraft)]
            self._case_indexes[aircraft] = self.case_module.CaseRetriever(ev, aliases, causal)
        return self._case_indexes[aircraft]

    def _v21(self, aircraft: str):
        if not self.evidence["aircraft_model"].eq(aircraft).any():
            raise ValueError(f"Unknown aircraft: {aircraft}")
        # The legacy V21 IDF statistics were fitted on the joint corpus; its
        # original query_scores applies the aircraft mask after scoring.
        if "joint_corpus" not in self._v21_indexes:
            self._v21_indexes["joint_corpus"] = (
                self.evidence, self.v21_module.build_indexes(self.evidence)
            )
        return self._v21_indexes["joint_corpus"]

    def _bounded_cache(self, question: str, aircraft: str) -> None:
        key = (question, aircraft)
        if key == self._last_question:
            return
        self._query_cache.clear()
        for index in self._case_indexes.values():
            for name, value in vars(index).items():
                if name.endswith("_cache") and isinstance(value, dict):
                    value.clear()
        self._last_question = key

    def retrieve(self, question: str, aircraft: str, method: str, k: int = 10) -> list[dict]:
        if not isinstance(question, str) or not isinstance(aircraft, str):
            raise TypeError("Only visible question and aircraft strings are accepted.")
        if method not in METHODS:
            raise ValueError(f"Unknown method: {method}")
        if k < 1:
            raise ValueError("k must be positive")
        self._bounded_cache(question, aircraft)
        route_note = "original controlled method"
        if method in V21_METHODS:
            ev, features = self._v21(aircraft)
            effective_k = min(k, int(ev["aircraft_model"].eq(aircraft).sum()))
            key = (question, aircraft, "v21_scores")
            if key not in self._query_cache:
                self._query_cache[key] = self._optimized_v21_scores(question, aircraft, ev, features)
            scores = self._query_cache[key]
            if method == "FD-GRAG-RC-V21-final":
                # The source template itself is visible. No benchmark track is passed.
                exact_visible = bool(self.v21_module.cue_segments(question)) and "Include evidence cues:" not in question
                if exact_visible:
                    routes = ["BM25/word", "FD-GRAG-RC-V21-cue-route", "FD-GRAG-RC-V21-package",
                              "FD-GRAG-RC-V21-exact", "Hybrid lexical", "TFIDF-char"]
                    quotas = [("BM25/word", 3), ("FD-GRAG-RC-V21-cue-route", 4), ("FD-GRAG-RC-V21-package", 3)]
                    route_note = "visible exact-cue marker"
                else:
                    routes = ["FD-GRAG-RC-V21-package", "FD-GRAG-RC-V21-cue-route", "Hybrid lexical",
                              "TFIDF-char", "BM25/word", "FD-GRAG-RC-V21-exact"]
                    quotas = [("FD-GRAG-RC-V21-package", 7), ("FD-GRAG-RC-V21-cue-route", 3)]
                    route_note = "visible broad/no-cue package route"
                idx = self.v21_module.route_union_indices(scores, routes, effective_k, quotas)
                rank_scores = [float(scores["FD-GRAG-RC-V21-package"][i]) for i in idx]
            else:
                idx = self.v21_module.top_indices(scores[method], effective_k)
                rank_scores = [float(scores[method][i]) for i in idx]
            selected = ev.iloc[idx]
        else:
            index = self._case(aircraft)
            ids, rank_scores, _ = index.retrieve(question, method, k=min(k, len(index.evidence)))
            selected = index.evidence.iloc[[index.id_to_idx[eid] for eid in ids]]
        rows = []
        for rank, ((_, row), score) in enumerate(zip(selected.iterrows(), rank_scores), 1):
            item = {c: row.get(c, "") for c in ONLINE_COLUMNS}
            # Evidence/source anchors remain raw and unchanged for post-hoc scoring.
            item.update({"method": method, "evidence_id": str(row["evidence_id"]),
                         "source_id": str(row.get("source_id") or row["evidence_id"]),
                         "rank": rank, "score": float(score), "route_note": route_note})
            rows.append(item)
        return rows

    def retrieve_all(self, question: str, aircraft: str, k: int = 10) -> dict[str, list[dict]]:
        return {method: self.retrieve(question, aircraft, method, k) for method in METHODS}


def build_index(project_root: str | Path, evidence: pd.DataFrame | None = None,
                aliases: pd.DataFrame | None = None,
                causal: pd.DataFrame | None = None) -> BaselineIndex:
    root = Path(project_root).resolve()
    if evidence is None:
        evidence = pd.read_parquet(root / "01_processed_corpus" / "v5" / "evidence_objects.parquet")
    if aliases is None:
        alias_path = root / "01_processed_corpus" / "v5" / "component_aliases.csv"
        aliases = pd.read_csv(alias_path).fillna("") if alias_path.exists() else pd.DataFrame()
    if causal is None:
        pieces = []
        for aircraft, slug in (("Boeing 737", "Boeing_737"), ("Cessna 172", "Cessna_172")):
            case = root / "Cases" / slug
            if case.exists():
                for path in case.glob("2 */*_causal_edges.csv"):
                    piece = pd.read_csv(path).fillna("")
                    piece["aircraft_model"] = aircraft
                    pieces.append(piece)
        causal = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    return BaselineIndex(root, evidence, aliases, causal)


def retrieve(index: BaselineIndex, question: str, aircraft: str,
             method: str, k: int = 10) -> list[dict]:
    return index.retrieve(question, aircraft, method, k)
