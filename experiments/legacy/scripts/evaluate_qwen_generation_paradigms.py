from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CASE_IDS = ["Boeing_737", "Cessna_172"]
OUT_DIR = ROOT / "04_results" / "qwen_generation_paradigm"
CHECKPOINT_DIR = OUT_DIR / "_checkpoints"
FIG_DIR = ROOT / "figures" / "Qwen_Paradigm_Comparison"
REPORT_DIR = ROOT / "reports"

DEFAULT_QWEN_MODELS = [
    {"scale": "small", "provider": "dashscope_openai_compatible", "model": "qwen-turbo"},
    {"scale": "medium", "provider": "dashscope_openai_compatible", "model": "qwen-plus"},
    {"scale": "large", "provider": "dashscope_openai_compatible", "model": "qwen-max"},
]

DEFAULT_OLLAMA_QWEN_MODELS = [
    {"scale": "qwen_3b_local", "provider": "ollama_openai_compatible", "model": "qwen2.5:3b"},
]

PARADIGM_METHODS = [
    ("qwen_direct_no_rag", ""),
    ("qwen_bm25_context", "BM25/word"),
    ("qwen_hybrid_context", "Hybrid lexical"),
    ("qwen_graphrag_context", "Controlled GraphRAG"),
    ("qwen_evidence_flow_context", "Evidence-flow baseline"),
    ("qwen_fd_grag_rc_context", "FD-GRAG-RC-V21-final"),
]

SOURCE_METRICS = [
    "exact_recall_10",
    "accepted_recall_10",
    "evidence_package_recall_10",
    "accepted_page_recall_10",
    "accepted_section_recall_10",
    "grounded_score",
    "hallucination_risk_proxy",
    "role_coverage_proxy",
]

SYSTEM_PROMPT = """You are an aviation maintenance evidence-grounded QA assistant.
Answer only from the provided evidence package when evidence is provided.
If the evidence is insufficient, say that the evidence is insufficient and request manual review.
Do not claim operational safety certification or real maintenance correctness.
Return a concise answer. Cite evidence ids when evidence ids are available."""


def table_dir(case_id: str) -> Path:
    case_root = ROOT / "Cases" / case_id
    test_dirs = [p for p in case_root.iterdir() if p.is_dir() and p.name.startswith("7")]
    if not test_dirs:
        raise FileNotFoundError(f"No test directory starting with 7 under {case_root}")
    return test_dirs[0] / "sota_evidence_chain_tables"


def case_figure_dir(case_id: str) -> Path:
    return ROOT / "Cases" / case_id / "8 最新图集"


def read_case_chain(case_id: str) -> pd.DataFrame:
    path = table_dir(case_id) / f"{case_id}_complete_sota_evidence_chains.csv"
    df = pd.read_csv(path).fillna("")
    df["case_id"] = case_id
    df["source_chain_csv"] = str(path)
    return df


def load_rows(case_ids: list[str], query_limit: int = 0) -> pd.DataFrame:
    frames = [read_case_chain(case_id) for case_id in case_ids]
    df = pd.concat(frames, ignore_index=True)
    wanted = [m for _, m in PARADIGM_METHODS if m]
    method_rows = df[df["method"].isin(wanted)].copy()
    direct_base = df[df["method"].eq("FD-GRAG-RC-V21-final")].copy()
    if query_limit and query_limit > 0:
        keep_queries = (
            direct_base[["case_id", "query_id"]]
            .drop_duplicates()
            .groupby("case_id", group_keys=False)
            .head(query_limit)
        )
        keys = set(map(tuple, keep_queries[["case_id", "query_id"]].to_records(index=False)))
        method_rows = method_rows[method_rows.apply(lambda r: (r["case_id"], r["query_id"]) in keys, axis=1)]
        direct_base = direct_base[direct_base.apply(lambda r: (r["case_id"], r["query_id"]) in keys, axis=1)]
    method_rows["paradigm"] = method_rows["method"].map({m: p for p, m in PARADIGM_METHODS if m})
    direct_base["paradigm"] = "qwen_direct_no_rag"
    direct_base["method"] = ""
    direct_base["method_display"] = "Qwen direct / no RAG"
    for col in [
        "top_evidence_ids",
        "top_evidence_scores",
        "top_evidence_pages",
        "top_evidence_roles",
        "top_evidence_components",
        "top_evidence_sections",
        "evidence_chain_top10",
    ]:
        if col in direct_base.columns:
            direct_base[col] = ""
    for col in SOURCE_METRICS:
        if col in direct_base.columns:
            direct_base[col] = 0.0
    return pd.concat([direct_base, method_rows], ignore_index=True)


def truncate_text(text: Any, max_chars: int) -> str:
    text = "" if pd.isna(text) else str(text)
    return text if len(text) <= max_chars else text[:max_chars] + "\n...[truncated]"


def split_semicolon(value: Any) -> list[str]:
    text = "" if pd.isna(value) else str(value)
    return [x.strip() for x in re.split(r"[;|]", text) if x.strip()]


def infer_visible_intent_slots(question: str) -> list[str]:
    """Infer required answer slots from visible question text only.

    This intentionally avoids benchmark-only query_type, gold evidence, accepted
    evidence, or package-slot labels. It is used only to structure the FD-GRAG-RC
    prompt contract for the generator.
    """

    q = (question or "").lower()
    slots: list[str] = []
    patterns = [
        ("symptom", r"symptom|fault|failure|malfunction|warning light|message|phenomenon|现象|故障|报码"),
        ("cause", r"cause|reason|why|failure mode|root|导致|原因|失效|故障模式"),
        ("excluded_cause", r"excluded|ruled out|not caused|not due|排除|不是|并非"),
        ("maintenance_action", r"action|procedure|repair|inspect|inspection|check|replace|troubleshoot|maintenance|处置|维修|检查|更换|排故"),
        ("threshold_or_condition", r"threshold|limit|condition|when|if|psi|rpm|degree|torque|限制|阈值|条件|范围"),
        ("warning_or_limitation", r"warning|caution|note|limitation|unsafe|do not|警告|注意|限制|禁止"),
        ("authority", r"manual|section|page|reference|authority|approved|手册|章节|页|依据"),
        ("component", r"component|system|ata|pack|valve|pump|engine|brake|gear|flap|aileron|组件|部件|系统|发动机|刹车|起落架"),
    ]
    for slot, pattern in patterns:
        if re.search(pattern, q, flags=re.I):
            slots.append(slot)
    if not slots:
        slots = ["symptom", "component", "maintenance_action", "authority"]
    if "maintenance_action" in slots and "authority" not in slots:
        slots.append("authority")
    return list(dict.fromkeys(slots))


def parse_evidence_chain(chain: Any) -> list[dict[str, str]]:
    text = "" if pd.isna(chain) else str(chain)
    if not text.strip():
        return []
    pattern = re.compile(
        r"(?:^|\n)\s*(\d+)\.\s+"
        r"(EVID[A-Z0-9_]+)\s+\|\s+"
        r"page=([^|]*)\|\s+role=([^|]*)\|\s+component=([^|]*)\|\s+section=([^|]*)\|\s*"
        r"(.*?)(?=\n\s*\d+\.\s+EVID[A-Z0-9_]+\s+\||\Z)",
        flags=re.S,
    )
    cards: list[dict[str, str]] = []
    for match in pattern.finditer(text):
        snippet = re.sub(r"\s+", " ", match.group(7)).strip()
        cards.append(
            {
                "rank": match.group(1).strip(),
                "evidence_id": match.group(2).strip(),
                "page": match.group(3).strip(),
                "role": match.group(4).strip(),
                "component": match.group(5).strip(),
                "section": match.group(6).strip(),
                "snippet": snippet,
            }
        )
    if cards:
        return cards

    # Fallback for legacy chains that preserve ids but not the full card format.
    ids = re.findall(r"EVID[A-Z0-9_]+", text)
    return [
        {
            "rank": str(i + 1),
            "evidence_id": eid,
            "page": "",
            "role": "",
            "component": "",
            "section": "",
            "snippet": "",
        }
        for i, eid in enumerate(ids[:10])
    ]


def role_bucket(card: dict[str, str]) -> str:
    role = (card.get("role", "") + " " + card.get("snippet", "")).lower()
    if any(x in role for x in ["warning", "caution", "limitation", "safety", "note"]):
        return "warning_or_limitation"
    if any(x in role for x in ["threshold", "limit", "torque", "condition", "psi", "rpm"]):
        return "threshold_or_condition"
    if any(x in role for x in ["authority", "manual", "section", "page", "reference"]):
        return "authority"
    if any(x in role for x in ["action", "inspection", "replace", "repair", "maintenance", "procedure", "check"]):
        return "maintenance_action"
    if any(x in role for x in ["excluded", "negative"]):
        return "excluded_cause"
    if any(x in role for x in ["cause", "failure", "fault"]):
        return "cause"
    if any(x in role for x in ["symptom", "finding"]):
        return "symptom"
    if card.get("component", "").strip():
        return "component"
    return "supporting_context"


def short_card(card: dict[str, str], max_snippet_chars: int = 300) -> str:
    snippet = truncate_text(card.get("snippet", ""), max_snippet_chars).replace("\n", " ")
    bits = [
        f"rank={card.get('rank', '')}",
        f"id={card.get('evidence_id', '')}",
        f"page={card.get('page', '')}",
        f"role={card.get('role', '')}",
        f"component={card.get('component', '')}",
    ]
    section = card.get("section", "")
    if section:
        bits.append(f"section={truncate_text(section, 110).replace(chr(10), ' ')}")
    return " | ".join(bits) + f" | snippet={snippet}"


def build_fd_grag_rc_contract(row: pd.Series, max_context_chars: int) -> str:
    question = str(row.get("question", ""))
    visible_slots = infer_visible_intent_slots(question)
    cards = parse_evidence_chain(row.get("evidence_chain_top10", ""))
    if not cards:
        return "FD-GRAG-RC evidence package contract: no retrieved evidence cards are available."

    grouped: dict[str, list[dict[str, str]]] = {}
    for card in cards:
        grouped.setdefault(role_bucket(card), []).append(card)
    covered = [slot for slot in visible_slots if slot in grouped]
    missing = [slot for slot in visible_slots if slot not in grouped]

    priority = [
        "maintenance_action",
        "cause",
        "excluded_cause",
        "threshold_or_condition",
        "warning_or_limitation",
        "authority",
        "component",
        "symptom",
        "supporting_context",
    ]
    selected: list[dict[str, str]] = []
    seen: set[str] = set()
    for bucket in priority:
        for card in grouped.get(bucket, [])[:2]:
            eid = card["evidence_id"]
            if eid not in seen:
                selected.append({**card, "bucket": bucket})
                seen.add(eid)
    for card in cards:
        if len(selected) >= 8:
            break
        eid = card["evidence_id"]
        if eid not in seen:
            selected.append({**card, "bucket": role_bucket(card)})
            seen.add(eid)

    lines = [
        "FD-GRAG-RC EVIDENCE PACKAGE CONTRACT",
        "1) Visible-query intent slots inferred without gold labels:",
        "   " + ", ".join(visible_slots),
        "2) Retrieved role coverage from evidence cards:",
        f"   covered={', '.join(covered) if covered else 'none'}",
        f"   missing_or_weak={', '.join(missing) if missing else 'none'}",
        "3) Use evidence IDs exactly as written. Every maintenance claim must cite at least one ID.",
        "4) If an asked-for action, threshold, warning, authority, or component slot is missing, answer conservatively and state that evidence is insufficient.",
        "",
        "Role-ordered evidence cards:",
    ]
    for card in selected:
        lines.append(f"- slot={card['bucket']} | {short_card(card)}")
    lines.extend(
        [
            "",
            "Required answer format:",
            "Grounded answer:",
            "- Symptom/condition:",
            "- Cause/failure mode:",
            "- Maintenance action:",
            "- Threshold/condition/warning/authority:",
            "- Evidence IDs used: EVID...",
            "If insufficient, replace the bullets with: Evidence is insufficient because ...; Evidence IDs checked: EVID...",
        ]
    )
    return truncate_text("\n".join(lines), max_context_chars)


def build_prompt(row: pd.Series, max_context_chars: int) -> list[dict[str, str]]:
    aircraft = str(row.get("aircraft_model", ""))
    question = str(row.get("question", ""))
    paradigm = str(row.get("paradigm", ""))
    if paradigm == "qwen_direct_no_rag":
        user = (
            f"Aircraft: {aircraft}\n"
            f"Question: {question}\n\n"
            "Answer from your parametric knowledge only. If you cannot ground the answer in known maintenance evidence, "
            "state the uncertainty and do not invent page-specific or procedure-specific details."
        )
    elif paradigm == "qwen_fd_grag_rc_context":
        context = build_fd_grag_rc_contract(row, max_context_chars)
        user = (
            f"Aircraft: {aircraft}\n"
            f"Question: {question}\n\n"
            f"{context}\n\n"
            "Follow the FD-GRAG-RC contract. Do not merge unrelated evidence cards into a stronger claim. "
            "Prefer a concise cited answer over a fluent uncited answer."
        )
    else:
        context = truncate_text(row.get("evidence_chain_top10", ""), max_context_chars)
        user = (
            f"Aircraft: {aircraft}\n"
            f"Question: {question}\n\n"
            f"Retrieved evidence package from paradigm `{paradigm}`:\n"
            f"{context}\n\n"
            "Use only this evidence package. Cite evidence ids such as EVID_... when making a maintenance statement. "
            "If the package lacks a required action, threshold, warning, component, or authority role, say that the evidence is insufficient."
        )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def parse_model_specs(args: argparse.Namespace) -> list[dict[str, str]]:
    if args.model_config:
        return pd.read_csv(args.model_config).fillna("").to_dict("records")
    if args.models:
        models = [x.strip() for x in args.models.split(",") if x.strip()]
        labels = [x.strip() for x in args.scale_labels.split(",") if x.strip()]
        if labels and len(labels) != len(models):
            raise ValueError("--scale-labels must match --models length")
        if not labels:
            labels = [f"qwen_{i + 1}" for i in range(len(models))]
        return [
            {"scale": scale, "provider": f"{args.backend}_openai_compatible", "model": model}
            for scale, model in zip(labels, models)
        ]
    if args.backend == "ollama":
        return DEFAULT_OLLAMA_QWEN_MODELS
    return DEFAULT_QWEN_MODELS


def write_model_template(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["scale", "provider", "model"])
        writer.writeheader()
        for row in DEFAULT_QWEN_MODELS:
            writer.writerow(row)
        for row in DEFAULT_OLLAMA_QWEN_MODELS:
            writer.writerow(row)


def call_openai_compatible(
    *,
    model: str,
    messages: list[dict[str, str]],
    api_key: str,
    base_url: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
) -> tuple[str, dict[str, Any]]:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"], data.get("usage", {})


def char_bigram_f1(pred: str, gold: str) -> float:
    pred = re.sub(r"\s+", "", pred or "")
    gold = re.sub(r"\s+", "", gold or "")
    if not pred or not gold:
        return 0.0
    p = Counter(pred[i : i + 2] for i in range(max(1, len(pred) - 1)))
    g = Counter(gold[i : i + 2] for i in range(max(1, len(gold) - 1)))
    overlap = sum((p & g).values())
    precision = overlap / max(1, sum(p.values()))
    recall = overlap / max(1, sum(g.values()))
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def bounded(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def evaluate_answer(answer: str, row: pd.Series) -> dict[str, Any]:
    answer = answer or ""
    available_ids = set(re.findall(r"EVID[A-Z0-9_]*", str(row.get("top_evidence_ids", ""))))
    cited_ids = set(re.findall(r"EVID[A-Z0-9_]*", answer))
    valid_ids = cited_ids & available_ids
    refusal_terms = [
        "insufficient",
        "not enough evidence",
        "manual review",
        "cannot determine",
        "无法",
        "证据不足",
        "不能确定",
        "人工复核",
    ]
    char_f1 = round(char_bigram_f1(answer, str(row.get("gold_answer", ""))), 4)
    citation_precision = len(valid_ids) / max(1, len(cited_ids)) if cited_ids else 0.0
    citation_coverage = 1.0 if valid_ids else 0.0
    source_grounded = to_float(row.get("grounded_score", 0.0))
    source_package = to_float(row.get("evidence_package_recall_10", 0.0))
    source_accepted = to_float(row.get("accepted_recall_10", 0.0))
    source_risk = to_float(row.get("hallucination_risk_proxy", 0.0))
    answer_evidence_grounded_score = bounded(
        0.22 * char_f1
        + 0.18 * citation_precision
        + 0.16 * citation_coverage
        + 0.18 * source_grounded
        + 0.18 * source_package
        + 0.10 * source_accepted
        - 0.12 * source_risk
    )
    return {
        "answer_chars": len(answer),
        "refusal_detected": any(t.lower() in answer.lower() for t in refusal_terms),
        "cited_evidence_count": len(cited_ids),
        "valid_cited_evidence_count": len(valid_ids),
        "citation_precision_proxy": citation_precision,
        "citation_coverage_proxy": citation_coverage,
        "char_bigram_f1_to_gold_proxy": char_f1,
        "answer_evidence_grounded_score_proxy": round(answer_evidence_grounded_score, 4),
    }


def write_status(
    *,
    status: str,
    reason: str,
    model_specs: list[dict[str, str]],
    prompt_rows: int,
    output_files: list[Path],
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Qwen Generation Paradigm Evaluation Status",
        "",
        f"- Status: {status}",
        f"- Reason: {reason}",
        f"- Prompt rows prepared: {prompt_rows}",
        f"- Created time: {datetime.now().isoformat(timespec='seconds')}",
        "- Base model family: Qwen through DashScope or Ollama OpenAI-compatible chat-completion API.",
        "- Paradigms: Qwen direct, Qwen + BM25, Qwen + Hybrid, Qwen + GraphRAG, Qwen + Evidence-flow, and Qwen + FD-GRAG-RC.",
        "- Data boundary: existing retrieval artifacts are read only; no benchmark, split, retrieval score, CSV/JSON/NPY semantics, or model training is changed.",
        "",
        "## Qwen model specs",
        "",
        "| Scale | Provider | Model |",
        "|---|---|---|",
    ]
    for spec in model_specs:
        lines.append(f"| {spec.get('scale','')} | {spec.get('provider','')} | {spec.get('model','')} |")
    lines += ["", "## Output files", ""]
    for p in output_files:
        lines.append(f"- `{p}`")
    lines += [
        "",
        "## Required run command",
        "",
        "Set `DASHSCOPE_API_KEY` or `QWEN_API_KEY`, then run:",
        "",
        "```powershell",
        "& <python> scripts/evaluate_qwen_generation_paradigms.py --run",
        "# or, for local Ollama Qwen 3B:",
        "& <python> scripts/evaluate_qwen_generation_paradigms.py --backend ollama --run",
        "```",
        "",
        "The bundled Python path in Codex desktop can be used if the system `python` alias is unavailable.",
    ]
    (REPORT_DIR / "qwen_generation_paradigm_status.md").write_text("\n".join(lines), encoding="utf-8")


def plot_summary(summary_path: Path, output_path: Path) -> Path | None:
    if not summary_path.exists():
        return None
    import matplotlib.pyplot as plt

    df = pd.read_csv(summary_path)
    if df.empty:
        return None

    metric_specs = [
        ("mean_source_accepted_recall_10", "Accepted\nRecall@10", "#4C78A8", "o"),
        ("mean_source_evidence_package_recall_10", "Package\nRecall@10", "#1B9E77", "s"),
        ("mean_source_grounded_score", "Grounded\nscore", "#2F855A", "D"),
        ("mean_source_role_coverage_proxy", "Role\ncoverage", "#F59E0B", "^"),
    ]
    available_specs = [m for m in metric_specs if m[0] in df.columns]
    if not available_specs:
        return None

    qwen_fd = df[df["paradigm"].eq("qwen_fd_grag_rc_context")]
    model = qwen_fd.iloc[0]["model"] if not qwen_fd.empty else df.iloc[0]["model"]
    data = df[df["model"].eq(model)].copy()
    order = [
        "qwen_bm25_context",
        "qwen_hybrid_context",
        "qwen_graphrag_context",
        "qwen_evidence_flow_context",
        "qwen_fd_grag_rc_context",
    ]
    data = data[data["paradigm"].isin(order)].copy()
    data["order"] = data["paradigm"].map({v: i for i, v in enumerate(order)})
    data = data.sort_values("order")
    labels = {
        "qwen_bm25_context": "Qwen + BM25",
        "qwen_hybrid_context": "Qwen + Hybrid",
        "qwen_graphrag_context": "Qwen + GraphRAG",
        "qwen_evidence_flow_context": "Qwen + Evidence-flow",
        "qwen_fd_grag_rc_context": "Qwen + FD-GRAG-RC",
    }
    fig, ax = plt.subplots(figsize=(9.6, 5.4), dpi=160)
    y_base = list(range(len(data)))
    offsets_by_count = {
        1: [0.0],
        2: [-0.16, 0.16],
        3: [-0.22, 0.0, 0.22],
        4: [-0.30, -0.10, 0.10, 0.30],
    }
    offsets = offsets_by_count.get(len(available_specs), [-0.30, -0.10, 0.10, 0.30])
    score_values = []
    for offset, (metric, label, color, marker) in zip(offsets, available_specs):
        values = pd.to_numeric(data[metric], errors="coerce").fillna(0.0)
        score_values.extend(values.tolist())
        ax.scatter(
            values,
            [y + offset for y in y_base],
            s=95,
            color=color,
            marker=marker,
            edgecolor="#222222",
            linewidth=0.65,
            label=label,
            zorder=3,
        )
        for yi, value, paradigm in zip(y_base, values, data["paradigm"]):
            if paradigm == "qwen_fd_grag_rc_context":
                ax.plot(
                    [min(value + 0.001, 1.001), 1.006],
                    [yi + offset, yi + offset],
                    color="#6B7280",
                    linewidth=0.65,
                    linestyle=(0, (2.0, 2.0)),
                    zorder=2,
                )
                ax.text(
                    1.007,
                    yi + offset,
                    f"{value:.3f}",
                    ha="left",
                    va="center",
                    fontsize=9.8,
                    color="#1F2937",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=0.15),
                )

    fd_rows = [i for i, p in enumerate(data["paradigm"]) if p == "qwen_fd_grag_rc_context"]
    for yi in fd_rows:
        ax.axhspan(yi - 0.42, yi + 0.42, color="#EAF2FB", zorder=0)

    xmin = max(0.0, min(score_values) - 0.018)
    xmax = 1.012 if max(score_values) > 0.985 else max(score_values) + 0.030
    ax.set_xlim(xmin, xmax)
    tick_start = round((int(xmin * 100) + 1) / 100, 2)
    ticks = []
    tick = tick_start
    while tick <= 1.0001:
        ticks.append(round(tick, 2))
        tick += 0.01
    ax.set_xticks(ticks)
    ax.set_ylim(-0.65, len(data) - 0.35)
    ax.set_yticks(y_base)
    ax.set_yticklabels([labels.get(p, p) for p in data["paradigm"]], fontsize=12.5)
    for tick, paradigm in zip(ax.get_yticklabels(), data["paradigm"]):
        if paradigm == "qwen_fd_grag_rc_context":
            tick.set_fontweight("bold")
            tick.set_color("#1F4E79")
    ax.set_xlabel("Evidence-package score", fontsize=13)
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.8, alpha=0.85)
    ax.tick_params(axis="x", labelsize=11, length=0)
    ax.tick_params(axis="y", length=0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ymin, ymax = ax.get_ylim()
    ax.annotate("", xy=(xmax, ymin), xytext=(xmin, ymin), arrowprops=dict(arrowstyle="->", lw=1.1, color="#333333"), clip_on=False)
    ax.annotate("", xy=(xmin, ymax), xytext=(xmin, ymin), arrowprops=dict(arrowstyle="->", lw=1.1, color="#333333"), clip_on=False)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=min(5, len(available_specs)),
        frameon=False,
        fontsize=10.5,
        handletextpad=0.5,
        columnspacing=1.1,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return output_path


def make_summary(response_records: list[dict[str, Any]]) -> pd.DataFrame:
    if not response_records:
        return pd.DataFrame()
    res = pd.DataFrame(response_records)
    return (
        res.groupby(["scale", "model", "paradigm", "source_method_display"], dropna=False)
        .agg(
            n=("query_id", "count"),
            ok_rate=("status", lambda s: (s == "ok").mean()),
            mean_latency_s=("latency_s", "mean"),
            mean_refusal_rate=("refusal_detected", "mean"),
            mean_citation_precision_proxy=("citation_precision_proxy", "mean"),
            mean_citation_coverage_proxy=("citation_coverage_proxy", "mean"),
            mean_char_bigram_f1_to_gold_proxy=("char_bigram_f1_to_gold_proxy", "mean"),
            mean_answer_evidence_grounded_score_proxy=("answer_evidence_grounded_score_proxy", "mean"),
            mean_source_exact_recall_10=("source_exact_recall_10", "mean"),
            mean_source_accepted_recall_10=("source_accepted_recall_10", "mean"),
            mean_source_grounded_score=("source_grounded_score", "mean"),
            mean_source_evidence_package_recall_10=("source_evidence_package_recall_10", "mean"),
            mean_source_accepted_page_recall_10=("source_accepted_page_recall_10", "mean"),
            mean_source_accepted_section_recall_10=("source_accepted_section_recall_10", "mean"),
            mean_source_role_coverage_proxy=("source_role_coverage_proxy", "mean"),
            mean_source_hallucination_risk_proxy=("source_hallucination_risk_proxy", "mean"),
        )
        .reset_index()
    )
    summary["mean_source_risk_control_proxy"] = 1.0 - summary["mean_source_hallucination_risk_proxy"]
    summary.loc[summary["paradigm"].eq("qwen_direct_no_rag"), "mean_source_risk_control_proxy"] = 0.0
    return summary


def write_summary_and_plot(response_records: list[dict[str, Any]], summary_path: Path, fig_path: Path) -> tuple[Path, Path | None]:
    summary = make_summary(response_records)
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    fig = plot_summary(summary_path, fig_path)
    return summary_path, fig


def write_case_summaries_and_plots(response_records: list[dict[str, Any]], case_ids: list[str]) -> list[Path]:
    if not response_records:
        return []
    res = pd.DataFrame(response_records)
    outputs: list[Path] = []
    for case_id in case_ids:
        case_records = res[res["case_id"].astype(str).eq(case_id)].to_dict("records")
        if not case_records:
            continue
        case_summary = OUT_DIR / f"{case_id}_qwen_generation_summary.csv"
        case_figure = case_figure_dir(case_id) / "qwen_3b_generation_paradigm_full.png"
        summary_path, fig_path = write_summary_and_plot(case_records, case_summary, case_figure)
        outputs.append(summary_path)
        if fig_path:
            outputs.append(fig_path)
    return outputs


def write_checkpoint(response_records: list[dict[str, Any]]) -> Path:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / "qwen_generation_responses_checkpoint.csv"
    pd.DataFrame(response_records).to_csv(checkpoint_path, index=False, encoding="utf-8-sig")
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=",".join(CASE_IDS))
    parser.add_argument("--backend", choices=["dashscope", "ollama"], default="dashscope")
    parser.add_argument("--models", default="")
    parser.add_argument("--scale-labels", default="")
    parser.add_argument("--model-config", default="")
    parser.add_argument("--query-limit-per-case", type=int, default=0)
    parser.add_argument("--only-paradigms", default="")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--max-context-chars", type=int, default=5200)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--flush-every", type=int, default=20)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    template = REPORT_DIR / "qwen_generation_models_template.csv"
    write_model_template(template)

    case_ids = [c.strip() for c in args.cases.split(",") if c.strip()]
    model_specs = parse_model_specs(args)
    rows = load_rows(case_ids, query_limit=args.query_limit_per_case)
    if args.only_paradigms:
        keep = {x.strip() for x in args.only_paradigms.split(",") if x.strip()}
        rows = rows[rows["paradigm"].isin(keep)].copy()
    prompt_records = []
    for spec in model_specs:
        for _, row in rows.iterrows():
            prompt_records.append(
                {
                    "case_id": row.get("case_id", ""),
                    "query_id": row.get("query_id", row.get("question_id", "")),
                    "question_id": row.get("question_id", ""),
                    "aircraft_model": row.get("aircraft_model", ""),
                    "query_type": row.get("query_type", ""),
                    "scale": spec.get("scale", ""),
                    "provider": spec.get("provider", ""),
                    "model": spec.get("model", ""),
                    "paradigm": row.get("paradigm", ""),
                    "source_method": row.get("method", ""),
                    "source_method_display": row.get("method_display", ""),
                    "messages_json": json.dumps(build_prompt(row, args.max_context_chars), ensure_ascii=False),
                    "source_chain_csv": row.get("source_chain_csv", ""),
                }
            )
    prompt_manifest = OUT_DIR / "qwen_generation_prompt_manifest.csv"
    pd.DataFrame(prompt_records).to_csv(prompt_manifest, index=False, encoding="utf-8-sig")
    output_files = [prompt_manifest, template]

    uses_ollama = any("ollama" in str(spec.get("provider", "")).lower() for spec in model_specs)
    if uses_ollama:
        api_key = os.environ.get("OLLAMA_API_KEY") or "ollama"
        base_url = os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434/v1"
    else:
        api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY") or ""
        base_url = os.environ.get("QWEN_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    if not args.run:
        write_status(
            status="prepared_not_run",
            reason="Prompt manifest prepared. Re-run with --run after setting DASHSCOPE_API_KEY or QWEN_API_KEY.",
            model_specs=model_specs,
            prompt_rows=len(prompt_records),
            output_files=output_files,
        )
        print(json.dumps({"status": "prepared_not_run", "prompt_rows": len(prompt_records)}, ensure_ascii=False, indent=2))
        return
    if (not uses_ollama) and (not api_key):
        write_status(
            status="not_run_missing_qwen_key",
            reason="DASHSCOPE_API_KEY and QWEN_API_KEY are not set in the current environment.",
            model_specs=model_specs,
            prompt_rows=len(prompt_records),
            output_files=output_files,
        )
        print(json.dumps({"status": "not_run_missing_qwen_key", "prompt_rows": len(prompt_records)}, ensure_ascii=False, indent=2))
        return

    row_lookup = {
        (str(r.get("case_id", "")), str(r.get("query_id", r.get("question_id", ""))), str(r.get("paradigm", "")), str(r.get("method", ""))): r
        for _, r in rows.iterrows()
    }
    responses_path = OUT_DIR / "qwen_generation_responses.csv"
    checkpoint_path = CHECKPOINT_DIR / "qwen_generation_responses_checkpoint.csv"
    completed: set[tuple[str, str, str, str, str]] = set()
    response_records = []
    if args.no_resume and checkpoint_path.exists():
        checkpoint_path.unlink()
    resume_path = None
    if not args.no_resume:
        if responses_path.exists():
            resume_path = responses_path
        elif checkpoint_path.exists():
            resume_path = checkpoint_path
    if resume_path:
        old = pd.read_csv(resume_path).fillna("")
        response_records = old.to_dict("records")
        for _, old_row in old.iterrows():
            completed.add(
                (
                    str(old_row.get("case_id", "")),
                    str(old_row.get("query_id", "")),
                    str(old_row.get("model", "")),
                    str(old_row.get("paradigm", "")),
                    str(old_row.get("source_method", "")),
                )
            )
    pending_records = [
        rec
        for rec in prompt_records
        if (
            str(rec["case_id"]),
            str(rec["query_id"]),
            str(rec["model"]),
            str(rec["paradigm"]),
            str(rec["source_method"]),
        )
        not in completed
    ]
    for rec_i, rec in enumerate(pending_records, 1):
        started = time.time()
        messages = json.loads(rec["messages_json"])
        try:
            answer, usage = call_openai_compatible(
                model=rec["model"],
                messages=messages,
                api_key=api_key,
                base_url=base_url,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
            )
            status = "ok"
            error = ""
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            answer = ""
            usage = {}
            status = "error"
            error = str(exc)
        row = row_lookup[(rec["case_id"], rec["query_id"], rec["paradigm"], rec["source_method"])]
        response_records.append(
            {
                **{k: rec[k] for k in ["case_id", "query_id", "question_id", "aircraft_model", "query_type", "scale", "provider", "model", "paradigm", "source_method", "source_method_display"]},
                **{f"source_{m}": to_float(row.get(m, 0.0)) for m in SOURCE_METRICS},
                "status": status,
                "error": error,
                "latency_s": round(time.time() - started, 3),
                "answer": answer,
                "usage_json": json.dumps(usage, ensure_ascii=False),
                **evaluate_answer(answer, row),
            }
        )
        if args.flush_every > 0 and rec_i % args.flush_every == 0:
            write_checkpoint(response_records)
    pd.DataFrame(response_records).to_csv(responses_path, index=False, encoding="utf-8-sig")
    summary_path, fig = write_summary_and_plot(
        response_records,
        OUT_DIR / "qwen_generation_summary.csv",
        FIG_DIR / "qwen_generation_paradigm_summary.png",
    )
    output_files.extend([responses_path, summary_path])
    if fig:
        output_files.append(fig)
    output_files.extend(write_case_summaries_and_plots(response_records, case_ids))
    write_status(
        status="completed",
        reason=(
            "Qwen generation calls completed through Ollama OpenAI-compatible endpoint."
            if uses_ollama
            else "Qwen generation calls completed through DashScope OpenAI-compatible endpoint."
        ),
        model_specs=model_specs,
        prompt_rows=len(prompt_records),
        output_files=output_files,
    )
    print(json.dumps({"status": "completed", "responses": len(response_records), "summary": str(summary_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
