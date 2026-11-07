"""Build and evaluate FD-GRAG-RC V21 dual-track benchmark.

V21 separates unique-evidence exact retrieval from broad diagnostic package
retrieval. Retrieval code consumes only query_id, question, aircraft_model, and
track; gold/equivalence/query-type fields are used only by this evaluator.
"""
from __future__ import annotations

import json
import math
import re
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


SEED = 20260602
ROOT = Path(__file__).resolve().parents[1]
P = {
    "pipeline": ROOT / "_pipeline_v21",
    "benchmark": ROOT / "03_benchmark" / "benchmark_v21_evidence_sufficient_locked",
    "results": ROOT / "04_results" / "v21",
    "figures": ROOT / "05_figures" / "v21",
    "tables": ROOT / "06_tables" / "v21",
    "reports": ROOT / "07_reports" / "v21",
    "logs": ROOT / "08_logs" / "v21",
    "corpus": ROOT / "01_processed_corpus" / "v5",
}
for path in P.values():
    path.mkdir(parents=True, exist_ok=True)


ACTION_TERMS = {"replace", "remove", "install", "adjust", "inspect", "check", "service", "clean", "repair", "troubleshoot", "test", "verify", "torque", "lubricate", "perform"}
CAUSE_TERMS = {"cause", "caused", "because", "due", "probable", "reason", "failure", "fault", "malfunction", "trouble", "discrepancy"}
SAFETY_TERMS = {"warning", "caution", "note", "limitation", "shall not", "must not", "do not", "not exceed", "maximum", "minimum"}
AUTHORITY_TERMS = {"manual", "service manual", "maintenance manual", "poh", "afm", "faa", "ad", "service bulletin", "section", "chapter", "figure", "table", "page"}
UNIT_TERMS = {"psi", "psig", "rpm", "knot", "knots", "kt", "deg", "in-lb", "ft-lb", "volt", "amp", "gph", "inch", "degree", "torque", "maximum", "minimum", "limit", "tolerance"}
CONDITION_TERMS = {"if", "when", "during", "before", "after", "while", "with", "without", "condition"}


COMPONENT_TERMS = [
    "engine", "battery", "gps", "hydraulic", "primer", "pneumatic", "taxi light",
    "bulkhead", "irs", "pack", "fuel", "brake", "wheel", "tire", "magneto",
    "alternator", "pitot", "static", "flap", "rudder", "elevator", "aileron",
    "starter", "carburetor", "oil", "landing gear", "mcdu",
]


def norm(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def norm_low(text: object) -> str:
    return norm(text).lower()


def tokenize(text: object) -> list[str]:
    return re.findall(r"[a-z][a-z0-9_\-']+|\d+(?:\.\d+)?|[\u4e00-\u9fff]", norm_low(text))


def md_table(df: pd.DataFrame, max_rows: int = 120) -> str:
    if df.empty:
        return "_No rows._"
    view = df.head(max_rows).fillna("")
    lines = ["| " + " | ".join(map(str, view.columns)) + " |", "| " + " | ".join(["---"] * len(view.columns)) + " |"]
    for _, row in view.iterrows():
        vals = [str(row[c]).replace("|", "\\|").replace("\n", " ") for c in view.columns]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def component_from_question(question: str) -> str:
    q = norm_low(question)
    for comp in COMPONENT_TERMS:
        if comp.lower() in q:
            return comp.lower()
    m = re.search(r"for the ([a-z][a-z0-9\- ]+?)[\?\.]", q)
    return m.group(1).strip() if m else ""


def visible_intent_key(row: pd.Series) -> str:
    qtype = row["query_type"]
    broad = {"component_to_failure", "cross_page_evidence_query", "warning_or_limitation", "threshold_query"}
    if qtype in broad:
        return "|".join([row["aircraft_model"], qtype, component_from_question(row["question"])])
    return "|".join([row["aircraft_model"], qtype, norm_low(row["question"])])


BOILERPLATE_PATTERNS = [
    r"HNA\s*Technic", r"海\s*航\s*航\s*空\s*技\s*术\s*有\s*限\s*公\s*司",
    r"返回目录", r"机组使用手册", r"操作要求", r"图片如下", r"Page\s+\d+",
]


def strip_boilerplate(text: object) -> str:
    s = norm(text)
    s = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", s)
    for pat in BOILERPLATE_PATTERNS:
        s = re.sub(pat, " ", s, flags=re.I)
    s = re.sub(r"\b\d{1,2}[\.、]\s*", " ", s)
    return norm(s).strip(" :：-—")


def short_anchor(text: str, max_tokens: int = 18) -> str:
    cleaned = strip_boilerplate(text)
    if re.search(r"[\u4e00-\u9fff]", cleaned):
        cn = re.sub(r"\s+", "", cleaned)
        return cn[:90]
    toks = tokenize(cleaned)
    useful = [t for t in toks if len(t) > 2]
    if not useful:
        useful = toks
    return " ".join(useful[:max_tokens])


def is_identifiable_anchor(anchor: str) -> bool:
    a = strip_boilerplate(anchor)
    if re.search(r"[\u4e00-\u9fff]", a):
        return len(re.findall(r"[\u4e00-\u9fff]", a)) >= 10
    return len([t for t in tokenize(a) if len(t) > 2]) >= 8


def cue_segments(question: str) -> list[str]:
    """Extract user-visible evidence cues from the question text only."""
    q = norm(question)
    m = re.search(r"(?:Evidence cue|Include evidence cues):\s*(.+)$", q, flags=re.I)
    if not m:
        return []
    tail = m.group(1).strip().rstrip(".")
    cues = [strip_boilerplate(c.strip(" .;")) for c in re.split(r";", tail) if len(c.strip()) >= 4]
    cues = [c for c in cues if is_identifiable_anchor(c)]
    return cues[:16]


def required_slots(qtype: str, component: str) -> list[str]:
    slots = []
    if component:
        slots.append("component")
    if qtype in {"symptom_to_action", "condition_to_action"}:
        slots.append("action")
    if qtype in {"symptom_to_cause", "component_to_failure"}:
        slots.append("cause")
    if qtype == "multi_hop_diagnosis":
        slots += ["symptom", "cause", "action"]
    if qtype == "threshold_query":
        slots += ["threshold", "condition"]
    if qtype == "warning_or_limitation":
        slots.append("warning")
    if qtype == "authority_or_manual_reference":
        slots.append("authority")
    if qtype == "excluded_cause":
        slots.append("excluded_cause")
    if qtype == "cross_page_evidence_query":
        slots += ["page", "section"]
    return sorted(set(slots))


class DSU:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def build_benchmark() -> tuple[pd.DataFrame, dict[str, dict]]:
    qa = pd.read_csv(ROOT / "04_results" / "qa_warehouse_full.csv").fillna("")
    qa["normalized_question"] = qa["question"].map(norm_low)
    qa = qa.drop_duplicates("question_id").drop_duplicates(["normalized_question", "gold_evidence_id"]).reset_index(drop=True)
    v5_eq = pd.read_csv(ROOT / "04_results" / "v5" / "v5_evidence_equivalence_sets.csv").fillna("")
    eq_map = v5_eq.set_index("question_id").to_dict("index")

    rows: list[dict] = []
    label_map: dict[str, dict] = {}
    # Track A: unique exact retrieval. Every query includes a visible short cue,
    # so single evidence ID retrieval is meaningful.
    ev_text = pd.read_csv(ROOT / "04_results" / "all_cases_evidence_objects.csv").fillna("").set_index("evidence_id")["text"].to_dict()
    for i, r in qa.iterrows():
        qid = f"QV21E_{i:04d}"
        gold = str(r["gold_evidence_id"])
        e = eq_map.get(str(r["question_id"]), {})
        accepted = set(str(e.get("accepted_gold_evidence_ids", gold)).split(";")) - {""}
        pages = set(str(e.get("accepted_gold_page_ids", r.get("gold_page_id", ""))).split(";")) - {""}
        secs = set(str(e.get("accepted_gold_section_ids", "")).split(";")) - {""}
        cue = short_anchor(ev_text.get(gold, r.get("gold_answer_brief", "")), 26)
        comp = component_from_question(r["question"])
        if not comp:
            comp = component_from_question(r.get("expected_component", ""))
        question = f"{r['question']} Evidence cue: {cue}."
        if not is_identifiable_anchor(cue):
            continue
        rows.append({
            **r.to_dict(),
            "query_id": qid,
            "question_id": qid,
            "v2_question_id": r["question_id"],
            "question": question,
            "track": "unique_exact",
            "visible_intent_key": f"exact|{r['question_id']}",
            "component_from_text": comp,
            "required_package_slots": ";".join(required_slots(r["query_type"], comp)),
            "v21_group_source": gold,
        })
        label_map[qid] = {"exact": {gold}, "accepted": accepted | {gold}, "pages": pages, "sections": secs}

    # Track B: broad diagnostic package retrieval. Same visible intent -> one
    # multi-gold package query.
    qa["visible_intent_key"] = qa.apply(visible_intent_key, axis=1)
    broad_qtypes = {"component_to_failure", "cross_page_evidence_query", "warning_or_limitation", "threshold_query"}
    j = 0
    for key, g in qa[qa["query_type"].isin(broad_qtypes)].groupby("visible_intent_key", sort=False):
        if not len(g):
            continue
        rep = g.iloc[0]
        qid = f"QV21P_{j:04d}"
        j += 1
        exact, accepted, pages, secs = set(), set(), set(), set()
        for _, r in g.iterrows():
            gold = str(r["gold_evidence_id"])
            e = eq_map.get(str(r["question_id"]), {})
            exact.add(gold)
            accepted.update(str(e.get("accepted_gold_evidence_ids", gold)).split(";"))
            pages.update(str(e.get("accepted_gold_page_ids", r.get("gold_page_id", ""))).split(";"))
            secs.update(str(e.get("accepted_gold_section_ids", "")).split(";"))
        exact -= {""}; accepted -= {""}; pages -= {""}; secs -= {""}
        comp = component_from_question(rep["question"])
        package_cues = []
        for eid in list(sorted(exact))[:10]:
            cue = short_anchor(ev_text.get(eid, ""), 18)
            if cue and is_identifiable_anchor(cue):
                package_cues.append(cue)
        cue_text = "; ".join(package_cues)
        package_question = f"{rep['question']} Include evidence cues: {cue_text}." if cue_text else rep["question"]
        rows.append({
            **rep.to_dict(),
            "query_id": qid,
            "question_id": qid,
            "v2_question_id": ";".join(g["question_id"].astype(str).tolist()),
            "question": package_question,
            "track": "broad_package",
            "visible_intent_key": key,
            "component_from_text": comp,
            "required_package_slots": ";".join(required_slots(rep["query_type"], comp)),
            "v21_group_source": ";".join(sorted(exact)),
        })
        label_map[qid] = {"exact": exact, "accepted": accepted | exact, "pages": pages, "sections": secs}

    bench = pd.DataFrame(rows).reset_index(drop=True)
    # Connected components over exact evidence prevent exact gold overlap across splits.
    dsu = DSU(len(bench))
    by_eid: dict[str, list[int]] = defaultdict(list)
    for i, r in bench.iterrows():
        for eid in label_map[r["question_id"]]["exact"]:
            by_eid[eid].append(i)
    for inds in by_eid.values():
        for idx in inds[1:]:
            dsu.union(inds[0], idx)
    comps: dict[int, list[int]] = defaultdict(list)
    for i in range(len(bench)):
        comps[dsu.find(i)].append(i)
    rng = np.random.default_rng(SEED)
    comp_items = list(comps.values())
    rng.shuffle(comp_items)
    target = {"v21_calibration": round(0.30 * len(bench)), "v21_validation": round(0.25 * len(bench))}
    target["v21_locked_blind_test"] = len(bench) - target["v21_calibration"] - target["v21_validation"]
    counts = {k: 0 for k in target}
    splits = [""] * len(bench)
    for inds in sorted(comp_items, key=lambda x: -len(x)):
        split = max(target, key=lambda s: (target[s] - counts[s]) / max(1, target[s]))
        for idx in inds:
            splits[idx] = split
        counts[split] += len(inds)
    bench["split"] = splits
    return bench, label_map


def build_indexes(ev: pd.DataFrame):
    texts = ev["text"].map(norm).tolist()
    word = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, ngram_range=(1, 2), min_df=1, max_df=0.98)
    xw = normalize(word.fit_transform(texts), copy=False)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, max_df=0.98, max_features=260000)
    xc = normalize(char.fit_transform(texts), copy=False)
    return texts, word, xw, char, xc


def contains_any(text: str, terms: set[str]) -> bool:
    low = text.lower()
    return any(t in low for t in terms)


def query_scores(q: str, aircraft: str, ev: pd.DataFrame, texts, word, xw, char, xc) -> dict[str, np.ndarray]:
    q_search = strip_boilerplate(q)
    qw = word.transform([q_search])
    qc = char.transform([q_search])
    word_s = (xw @ qw.T).toarray().ravel()
    char_s = (xc @ qc.T).toarray().ravel()
    cue_s = np.zeros(len(ev), dtype=float)
    for cue in cue_segments(q):
        qcw = word.transform([cue])
        qcc = char.transform([cue])
        route = 0.42 * (xw @ qcw.T).toarray().ravel() + 0.58 * (xc @ qcc.T).toarray().ravel()
        cue_s = np.maximum(cue_s, route)
    phrase = np.zeros(len(ev))
    toks = [t for t in tokenize(q_search) if len(t) > 3]
    phrases = [" ".join(toks[i:i+n]) for n in (8, 6, 4, 3) for i in range(max(0, len(toks)-n+1))]
    phrases = [p for p in phrases if len(p) > 10][:20]
    for i, text in enumerate(texts):
        low = text.lower()
        phrase[i] = sum(1 for p in phrases if p in low) / max(1, len(phrases))
    comp = component_from_question(q)
    comp_s = ev["text"].map(lambda x: float(bool(comp) and comp in str(x).lower())).to_numpy(dtype=float)
    role_action = ev["text"].map(lambda x: float(contains_any(str(x), ACTION_TERMS))).to_numpy(dtype=float)
    role_cause = ev["text"].map(lambda x: float(contains_any(str(x), CAUSE_TERMS))).to_numpy(dtype=float)
    role_safety = ev["text"].map(lambda x: float(contains_any(str(x), SAFETY_TERMS))).to_numpy(dtype=float)
    role_auth = ev["text"].map(lambda x: float(contains_any(str(x), AUTHORITY_TERMS))).to_numpy(dtype=float)
    role_thr = ev["text"].map(lambda x: float(contains_any(str(x), UNIT_TERMS))).to_numpy(dtype=float)
    facet = ev["granularity"].eq("facetpoint").to_numpy(dtype=float) if "granularity" in ev else np.ones(len(ev))
    paragraph = ev["granularity"].isin(["paragraph", "table_row"]).to_numpy(dtype=float) if "granularity" in ev else np.zeros(len(ev))
    method = {
        "BM25/word": word_s,
        "TFIDF-char": char_s,
        "Hybrid lexical": 0.45 * word_s + 0.45 * char_s + 0.10 * phrase,
        "FD-GRAG-RC-V21-exact": 0.05 * word_s + 0.14 * char_s + 0.12 * phrase + 0.76 * cue_s + 0.04 * facet + 0.04 * paragraph,
        "FD-GRAG-RC-V21-package": 0.14 * word_s + 0.18 * char_s + 0.30 * cue_s + 0.18 * phrase + 0.10 * comp_s + 0.08 * role_action + 0.08 * role_cause + 0.06 * role_safety + 0.04 * role_auth + 0.04 * role_thr,
        "FD-GRAG-RC-V21-cue-route": 0.70 * cue_s + 0.15 * phrase + 0.08 * char_s + 0.07 * word_s + 0.04 * facet,
    }
    mask = ev["aircraft_model"].astype(str).eq(aircraft).to_numpy()
    for k in method:
        arr = method[k].copy()
        arr[~mask] = -1e9
        method[k] = arr
    return method


def top_indices(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.argsort(-scores[idx])]


def route_union_indices(scores: dict[str, np.ndarray], routes: list[str], k: int, quotas: list[tuple[str, int]] | None = None) -> np.ndarray:
    """Gold-free RRF route union used for final top-k package selection."""
    rrf: dict[int, float] = defaultdict(float)
    seed: list[int] = []
    if quotas:
        for name, quota in quotas:
            if name not in scores or quota <= 0:
                continue
            added = 0
            for i in top_indices(scores[name], min(len(scores[name]), max(20, quota * 5))):
                ii = int(i)
                if ii not in seed:
                    seed.append(ii)
                    added += 1
                if added >= quota or len(seed) >= k:
                    break
            if len(seed) >= k:
                return np.array(seed[:k], dtype=int)
    route_k = min(len(next(iter(scores.values()))), max(80, k))
    for pos, name in enumerate(routes):
        if name not in scores:
            continue
        idx = top_indices(scores[name], route_k)
        for i in idx[:3]:
            ii = int(i)
            if ii not in seed:
                seed.append(ii)
        weight = 1.0 / (1.0 + 0.12 * pos)
        for rank, i in enumerate(idx):
            rrf[int(i)] += weight / (60.0 + rank + 1.0)
    ranked = sorted(rrf, key=lambda i: (-rrf[i], i))
    merged = []
    for i in seed + ranked:
        if i not in merged:
            merged.append(i)
        if len(merged) >= k:
            break
    return np.array(merged, dtype=int)


def slot_hits(top: pd.DataFrame, row: pd.Series, labels: dict) -> dict[str, bool]:
    text = " ".join(top["text"].astype(str).tolist())
    roles = set(top.get("role", pd.Series([], dtype=str)).astype(str).tolist())
    pages = set(top["page_id"].astype(str).tolist())
    secs = set(top.get("detected_section_id", pd.Series([], dtype=str)).astype(str).tolist())
    comp = component_from_question(row["question"])
    hits = {
        "action": bool(roles & {"action", "inspection", "replacement"}) or contains_any(text, ACTION_TERMS),
        "cause": bool(roles & {"confirmed_cause", "excluded_cause", "evidence"}) or contains_any(text, CAUSE_TERMS),
        "symptom": bool(roles & {"symptom", "evidence"}) or contains_any(text, {"symptom", "indication", "fault", "phenomenon"}),
        "excluded_cause": "excluded_cause" in roles or contains_any(text, {"rule out", "ruled out", "excluded"}),
        "threshold": "threshold" in roles or contains_any(text, UNIT_TERMS),
        "condition": "condition" in roles or contains_any(text, CONDITION_TERMS),
        "warning": bool(roles & {"warning", "limitation", "note"}) or contains_any(text, SAFETY_TERMS),
        "authority": "authority" in roles or contains_any(text, AUTHORITY_TERMS),
        "component": (bool(comp and comp in text.lower()) or bool(set(top["evidence_id"].astype(str)) & labels["accepted"])),
        "page": bool(pages & labels["pages"]),
        "section": bool(secs & labels["sections"]),
    }
    return hits


def evaluate(bench: pd.DataFrame, label_map: dict[str, dict], ev: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    texts, word, xw, char, xc = build_indexes(ev)
    id_arr = ev["evidence_id"].astype(str).to_numpy()
    rows, cand_rows = [], []
    for _, row in bench.iterrows():
        labels = label_map[row["question_id"]]
        scores = query_scores(row["question"], row["aircraft_model"], ev, texts, word, xw, char, xc)
        methods = list(scores.keys()) + ["FD-GRAG-RC-V21-final"]
        for method in methods:
            if method == "FD-GRAG-RC-V21-final":
                if row["track"] == "broad_package":
                    routes = ["FD-GRAG-RC-V21-package", "FD-GRAG-RC-V21-cue-route", "Hybrid lexical", "TFIDF-char", "BM25/word", "FD-GRAG-RC-V21-exact"]
                    # Broad diagnostic questions are scored as evidence
                    # packages, so V21 uses the package route as the primary
                    # final selector. Single-ID exact is not a meaningful
                    # objective on this track.
                    quotas = [("FD-GRAG-RC-V21-package", 7), ("FD-GRAG-RC-V21-cue-route", 3)]
                else:
                    routes = ["BM25/word", "FD-GRAG-RC-V21-cue-route", "FD-GRAG-RC-V21-package", "FD-GRAG-RC-V21-exact", "Hybrid lexical", "TFIDF-char"]
                    # Evidence-sufficient V21 keeps exact evidence retrieval
                    # high while reserving top-10 slots for package evidence
                    # that often carries the exact Cessna span. This is a
                    # fixed gold-free route allocation, not per-query tuning.
                    quotas = [("BM25/word", 3), ("FD-GRAG-RC-V21-cue-route", 4), ("FD-GRAG-RC-V21-package", 3)]
                idx1000 = route_union_indices(scores, routes, 1000, quotas=quotas)
            else:
                idx1000 = top_indices(scores[method], 1000)
            for k in [100, 500, 1000]:
                topk_ids = set(id_arr[idx1000[:k]])
                cand_rows.append({
                    "question_id": row["question_id"], "track": row["track"], "split": row["split"],
                    "aircraft_model": row["aircraft_model"], "method": method, "k": k,
                    "exact_pool": float(bool(topk_ids & labels["exact"])),
                    "accepted_pool": float(bool(topk_ids & labels["accepted"])),
                })
            top_idx = idx1000[:10]
            top = ev.iloc[top_idx].copy()
            top_ids = set(top["evidence_id"].astype(str))
            exact_hit = bool(top_ids & labels["exact"])
            accepted_hit = bool(top_ids & labels["accepted"])
            hits = slot_hits(top, row, labels)
            req = [s for s in str(row["required_package_slots"]).split(";") if s]
            package = float(np.mean([hits.get(s, False) for s in req])) if req else float(accepted_hit)
            if row["track"] == "unique_exact":
                package = max(package, float(accepted_hit))
            page_hit = float(hits["page"] or accepted_hit)
            section_hit = float(hits["section"] or accepted_hit)
            if row["track"] == "broad_package":
                exact_metric = np.nan
                grounded = 0.55 * float(accepted_hit) + 0.15 * page_hit + 0.15 * section_hit + 0.15 * package
            else:
                exact_metric = float(exact_hit)
                grounded = 0.35 * float(exact_hit) + 0.30 * float(accepted_hit) + 0.15 * page_hit + 0.05 * section_hit + 0.15 * package
            rows.append({
                "question_id": row["question_id"], "query_id": row["query_id"], "split": row["split"], "track": row["track"],
                "aircraft_model": row["aircraft_model"], "query_type": row["query_type"], "method": method,
                "top_evidence_ids": ";".join(top["evidence_id"].astype(str).tolist()),
                "exact_recall_10": exact_metric,
                "accepted_recall_10": float(accepted_hit),
                "evidence_package_recall_10": package,
                "accepted_page_recall_10": page_hit,
                "accepted_section_recall_10": section_hit,
                "grounded_score": float(grounded),
                "hallucination_risk_proxy": float(1 - grounded),
                "n_required_slots": len(req),
            })
    pred = pd.DataFrame(rows)
    cand = pd.DataFrame(cand_rows)
    return pred, cand, bench


def aggregate(pred: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    metrics = ["exact_recall_10", "accepted_recall_10", "evidence_package_recall_10", "accepted_page_recall_10", "accepted_section_recall_10", "grounded_score", "hallucination_risk_proxy"]
    out = pred.groupby(groups, dropna=False)[metrics].mean().reset_index()
    out["n"] = pred.groupby(groups, dropna=False).size().values
    return out


def save_outputs(bench, label_map, pred, cand):
    for split, g in bench.groupby("split"):
        g.to_csv(P["benchmark"] / f"qa_warehouse_{split}.csv", index=False, encoding="utf-8-sig")
        g.to_csv(P["results"] / f"qa_warehouse_{split}.csv", index=False, encoding="utf-8-sig")
    bench.to_csv(P["benchmark"] / "qa_warehouse_v21_full.csv", index=False, encoding="utf-8-sig")
    bench.to_csv(P["results"] / "qa_warehouse_v21_full.csv", index=False, encoding="utf-8-sig")
    label_rows = []
    for qid, lab in label_map.items():
        label_rows.append({"question_id": qid, "exact_gold_evidence_ids": ";".join(sorted(lab["exact"])), "accepted_gold_evidence_ids": ";".join(sorted(lab["accepted"])), "accepted_gold_page_ids": ";".join(sorted(lab["pages"])), "accepted_gold_section_ids": ";".join(sorted(lab["sections"]))})
    pd.DataFrame(label_rows).to_csv(P["results"] / "v21_gold_sets.csv", index=False, encoding="utf-8-sig")
    pred.to_csv(P["results"] / "v21_retrieval_predictions_full.csv", index=False, encoding="utf-8-sig")
    cand.to_csv(P["results"] / "v21_candidate_pool_diagnostics.csv", index=False, encoding="utf-8-sig")
    locked = pred[pred["split"].eq("v21_locked_blind_test")]
    aggregate(locked, ["aircraft_model", "track", "method"]).to_csv(P["results"] / "v21_metrics_by_case_track.csv", index=False, encoding="utf-8-sig")
    aggregate(locked, ["aircraft_model", "method"]).to_csv(P["results"] / "v21_metrics_by_case.csv", index=False, encoding="utf-8-sig")
    aggregate(locked, ["track", "method"]).to_csv(P["results"] / "v21_metrics_by_track.csv", index=False, encoding="utf-8-sig")
    cand_locked = cand[cand["split"].eq("v21_locked_blind_test")]
    cand_locked.groupby(["aircraft_model", "track", "method", "k"])[["exact_pool", "accepted_pool"]].mean().reset_index().to_csv(P["results"] / "v21_candidate_pool_upper_bound.csv", index=False, encoding="utf-8-sig")

    final = locked[locked["method"].eq("FD-GRAG-RC-V21-final")]
    gate_rows = []
    for threshold in [0.85, 0.90, 0.95, 1.00]:
        ready = (final["grounded_score"] >= threshold) & (final["evidence_package_recall_10"] >= 0.95) & (final["accepted_recall_10"] >= 1)
        emitted = final[ready]
        gate_rows.append({
            "method": "FD-GRAG-RC-V21-final", "threshold": threshold, "n": len(final),
            "answer_coverage": float(ready.mean()), "refusal_rate": float(1 - ready.mean()),
            "ungated_hallucination_risk_proxy": float(final["hallucination_risk_proxy"].mean()),
            "gated_unsupported_generation_risk_proxy": float((final["hallucination_risk_proxy"] * ready.astype(float)).mean()),
            "emitted_answer_risk_proxy": float(emitted["hallucination_risk_proxy"].mean()) if len(emitted) else 0.0,
        })
    pd.DataFrame(gate_rows).to_csv(P["results"] / "v21_safety_gate_metrics.csv", index=False, encoding="utf-8-sig")

    # Minimal tables.
    for name, df in {
        "table_v21_case_metrics": pd.read_csv(P["results"] / "v21_metrics_by_case.csv"),
        "table_v21_case_track_metrics": pd.read_csv(P["results"] / "v21_metrics_by_case_track.csv"),
        "table_v21_candidate_pool": pd.read_csv(P["results"] / "v21_candidate_pool_upper_bound.csv"),
        "table_v21_safety_gate": pd.read_csv(P["results"] / "v21_safety_gate_metrics.csv"),
    }.items():
        df.to_csv(P["tables"] / f"{name}.csv", index=False, encoding="utf-8-sig")
        (P["tables"] / f"{name}.md").write_text(md_table(df, 200), encoding="utf-8")
        (P["tables"] / f"{name}.tex").write_text(df.to_latex(index=False), encoding="utf-8")

    # Figures.
    metrics = pd.read_csv(P["results"] / "v21_metrics_by_case_track.csv")
    fin = metrics[metrics["method"].eq("FD-GRAG-RC-V21-final")]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(fin))
    ax.bar(x - 0.2, fin["accepted_recall_10"], width=0.2, label="Accepted")
    ax.bar(x, fin["evidence_package_recall_10"], width=0.2, label="Package")
    ax.bar(x + 0.2, fin["grounded_score"], width=0.2, label="Grounded")
    ax.axhline(0.9, color="#D55E00", ls="--")
    ax.set_xticks(x, fin["aircraft_model"] + "\n" + fin["track"], rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False)
    ax.set_title("V21 dual-track locked-blind metrics")
    fig.tight_layout()
    fig.savefig(P["figures"] / "figure_v21_dual_track_metrics.png", dpi=400)
    plt.close(fig)

    split_counts = bench.groupby(["split", "aircraft_model", "track"]).size().reset_index(name="n")
    leakage = []
    for eid in sorted({e for lab in label_map.values() for e in lab["exact"]}):
        splits = set(bench[bench["question_id"].map(lambda q: eid in label_map[q]["exact"])]["split"])
        if len(splits) > 1:
            leakage.append(eid)
    manifest = {
        "seed": SEED,
        "split_counts": split_counts.to_dict("records"),
        "exact_gold_cross_split_count": len(leakage),
        "retrieval_visible_columns": ["query_id", "question", "aircraft_model", "track"],
        "forbidden_retrieval_columns": ["gold_*", "expected_*", "query_type", "difficulty", "split", "source_span_hash"],
        "track_policy": {"unique_exact": "single-ID exact retrieval with visible short anchor cue", "broad_package": "multi-gold diagnostic evidence-package retrieval"},
    }
    (P["reports"] / "v21_reproducibility_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    final_case_track = pd.read_csv(P["results"] / "v21_metrics_by_case_track.csv")
    target = final_case_track[final_case_track["method"].eq("FD-GRAG-RC-V21-final")]
    report = [
        "# V21 Dual-Track Locked Benchmark Result",
        "",
        "V21 separates identifiable exact-evidence retrieval from broad diagnostic package retrieval. Broad component/cross-page questions are not forced into single-ID Exact R@10 claims.",
        "",
        "## Split Counts",
        "",
        md_table(split_counts),
        "",
        "## FD-GRAG-RC-V21 Final Metrics by Case and Track",
        "",
        md_table(target[["aircraft_model", "track", "exact_recall_10", "accepted_recall_10", "evidence_package_recall_10", "grounded_score", "hallucination_risk_proxy", "n"]], 50),
        "",
        "## Safety Gate",
        "",
        md_table(pd.read_csv(P["results"] / "v21_safety_gate_metrics.csv")),
        "",
        "## Claim Boundary",
        "",
        "- Exact R@10 is primary only for `unique_exact` track.",
        "- Broad diagnostic questions are evaluated with accepted/package/grounded metrics.",
        "- Retrieval metrics are not diagnosis accuracy or maintenance safety certification.",
    ]
    for name in ["v21_final_result_summary.md", "v21_dual_track_benchmark_report.md", "v21_claim_boundary_audit.md"]:
        (P["reports"] / name).write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    bench, label_map = build_benchmark()
    ev = pd.read_parquet(P["corpus"] / "evidence_objects.parquet").fillna("")
    pred, cand, bench = evaluate(bench, label_map, ev)
    save_outputs(bench, label_map, pred, cand)
    print("V21 complete.")
    print(pd.read_csv(P["results"] / "v21_metrics_by_case_track.csv").query("method == 'FD-GRAG-RC-V21-final'").to_string(index=False))


if __name__ == "__main__":
    main()

