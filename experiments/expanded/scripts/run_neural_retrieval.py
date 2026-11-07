"""Offline neural retrieval and real mMARCO reranking with matched CE budgets.

Neural weights and tokenizers are local; no SVD substitutes for embeddings.
Mean-pooled multilingual MiniLM is encoded exactly as its local pooling config.
Labels enter only evaluate() after all candidate pools and rankings are frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from contextlib import nullcontext
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from run_expanded_retrieval import ROOT, OUT, BM25, GranularRetriever, evaluate, norm
from baseline_adapter import _load_v21_functions


DENSE_PATH = Path(r"${USER_HOME}\.cache\huggingface\hub\models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2\snapshots\e8f8c211226b894fcb81acc59f3b34ba3efd5f42")
CE_PATH = Path(r"${USER_HOME}\.cache\huggingface\hub\models--cross-encoder--mmarco-mMiniLMv2-L12-H384-v1\snapshots\1427fd652930e4ba29e8149678df786c240d8825")
METHODS = (
    "Neural dense (multilingual MiniLM)",
    "Neural dense + mMARCO CE@30",
    "Neural lexical+dense RRF",
    "Neural dense + mMARCO CE@50",
    "IntentRAG multi + mMARCO CE@50",
)


def sha_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def top(scores: np.ndarray, eligible: np.ndarray, k: int) -> np.ndarray:
    indices = np.flatnonzero(eligible)
    return indices[np.argsort(-scores[indices], kind="stable")[:min(k, len(indices))]]


def rank_values(scores: np.ndarray, eligible: np.ndarray, pool: int = 120, constant: float = 60) -> np.ndarray:
    result = np.zeros(len(scores), dtype=np.float32)
    indices = top(scores, eligible, pool)
    result[indices] = 1.0 / (constant + np.arange(1, len(indices) + 1))
    return result


class LocalNeural:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.device = torch.device(args.device)
        self.dense_tokenizer = AutoTokenizer.from_pretrained(args.dense_path, local_files_only=True)
        self.dense = AutoModel.from_pretrained(args.dense_path, local_files_only=True).to(self.device).eval()
        self.ce_tokenizer = AutoTokenizer.from_pretrained(args.ce_path, local_files_only=True)
        self.ce = AutoModelForSequenceClassification.from_pretrained(args.ce_path, local_files_only=True).to(self.device).eval()

    def precision_context(self) -> Any:
        return torch.autocast(device_type="cuda", dtype=torch.float16) if self.args.precision == "fp16" else nullcontext()

    @torch.inference_mode()
    def encode(self, texts: list[str]) -> np.ndarray:
        encoded = self.dense_tokenizer(texts, padding=True, truncation=True,
            max_length=self.args.dense_max_length, return_tensors="pt", return_token_type_ids=False)
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with self.precision_context():
            hidden = self.dense(**encoded).last_hidden_state.float()
        mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        # Cosine-normalization preserves the original mean-pooling embedding.
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def rerank(self, query: str, texts: list[str]) -> np.ndarray:
        chunks = []
        for start in range(0, len(texts), self.args.ce_batch_size):
            batch = texts[start:start + self.args.ce_batch_size]
            encoded = self.ce_tokenizer([query] * len(batch), batch, padding=True,
                truncation=True, max_length=self.args.ce_max_length, return_tensors="pt",
                return_token_type_ids=False)
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with self.precision_context():
                chunks.append(self.ce(**encoded).logits[:, 0].float().cpu().numpy())
        return np.concatenate(chunks).astype(np.float32) if chunks else np.asarray([], dtype=np.float32)


def corpus_embeddings(ev: pd.DataFrame, neural: LocalNeural, args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    texts = ev.text.map(norm).tolist()
    unique, inverse = np.unique(np.asarray(texts, dtype=object), return_inverse=True)
    key = hashlib.sha256(("\n".join(texts) + str(args.dense_path) + str(args.dense_max_length) + args.device + args.precision).encode("utf-8")).hexdigest()[:20]
    cache_dir = OUT / "models" / "neural_corpus_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"embeddings_{key}.npy"
    state_path = cache_dir / f"unique_{key}.state.json"
    partial_path = cache_dir / f"unique_{key}.partial.npy"
    info = {"cache_key": key, "cache_path": str(path), "n_corpus": len(texts), "n_unique_texts": len(unique), "device": args.device, "precision": args.precision}
    if path.exists():
        result = np.load(path, mmap_mode="r")
        if result.shape != (len(ev), 384):
            raise ValueError(f"Embedding shape mismatch: {result.shape}")
        info["cache_hit"] = True
        return result, info
    info["cache_hit"] = False
    start_index = 0
    if partial_path.exists() and state_path.exists():
        start_index = int(json.loads(state_path.read_text(encoding="utf-8"))["completed"])
        array = np.lib.format.open_memmap(partial_path, mode="r+")
    else:
        array = np.lib.format.open_memmap(partial_path, mode="w+", dtype=np.float32, shape=(len(unique), 384))
    start_time = time.perf_counter()
    for start in range(start_index, len(unique), args.dense_batch_size):
        end = min(start + args.dense_batch_size, len(unique))
        array[start:end] = neural.encode(unique[start:end].tolist())
        array.flush()
        state_path.write_text(json.dumps({"completed": end, "total": len(unique), "cache_key": key}), encoding="utf-8")
        if start == start_index or end % (args.dense_batch_size * 20) == 0 or end == len(unique):
            print(json.dumps({"phase": "corpus_embedding", "completed": end, "total": len(unique), "elapsed_s": round(time.perf_counter()-start_time, 2)}, ensure_ascii=False), flush=True)
    np.save(path, np.asarray(array)[inverse])
    info["encoding_elapsed_s"] = time.perf_counter() - start_time
    return np.load(path, mmap_mode="r"), info


def registry(args: argparse.Namespace) -> list[dict[str, Any]]:
    return [
        {"method": METHODS[0], "candidate_pool": args.candidate_pool, "ce_pool": 0, "note": "Cosine multilingual MiniLM, mean pooling, full local model weights."},
        {"method": METHODS[1], "candidate_pool": args.candidate_pool, "ce_pool": args.protocol_ce_pool, "note": "Dense top50; real mMARCO reranks first30, manuscript protocol."},
        {"method": METHODS[2], "candidate_pool": args.candidate_pool*2, "ce_pool": 0, "note": "Okapi BM25 + true dense reciprocal-rank fusion; constant60; top50 each."},
        {"method": METHODS[3], "candidate_pool": args.candidate_pool, "ce_pool": args.matched_ce_pool, "note": "Dense top50; shared real mMARCO CE50 budget."},
        {"method": METHODS[4], "candidate_pool": args.candidate_pool, "ce_pool": args.matched_ce_pool, "note": "Entropy-weighted document detail/page/section/profile ranks + dense RRF; top50 candidates; same mMARCO CE50 budget."},
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=OUT / "data" / "canonical_questions.json")
    parser.add_argument("--dense-path", type=Path, default=DENSE_PATH)
    parser.add_argument("--ce-path", type=Path, default=CE_PATH)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--device", choices=("cpu", "cuda", "directml"), default="cpu")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--dense-batch-size", type=int, default=64)
    parser.add_argument("--ce-batch-size", type=int, default=16)
    parser.add_argument("--dense-max-length", type=int, default=128)
    parser.add_argument("--ce-max-length", type=int, default=384)
    parser.add_argument("--candidate-pool", type=int, default=50)
    parser.add_argument("--protocol-ce-pool", type=int, default=30)
    parser.add_argument("--matched-ce-pool", type=int, default=50)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable; refusing silent CPU timing fallback.")
    if args.precision == "fp16" and args.device != "cuda":
        parser.error("FP16 path requires CUDA.")
    if args.candidate_pool != 50 or args.protocol_ce_pool != 30 or args.matched_ce_pool != 50:
        parser.error("Frozen method labels require candidate50, protocol CE30, matched CE50.")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    for name in ("results", "reports", "models"):
        (OUT / name).mkdir(exist_ok=True)
    for directory in (args.dense_path, args.ce_path):
        for required in ("config.json", "tokenizer.json", "model.safetensors"):
            if not (directory / required).is_file():
                raise FileNotFoundError(directory / required)
    source = ROOT / "01_processed_corpus" / "v5" / "evidence_objects.parquet"
    source_hash_at_start = sha_file(source)
    question_hash_at_start = sha_file(args.questions)
    full_ev = pd.read_parquet(source).fillna("").drop_duplicates("evidence_id").reset_index(drop=True)
    evmap = full_ev.set_index("evidence_id").to_dict("index")
    ev = full_ev
    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    if args.smoke:
        # Document-only subsampling, never selecting corpus rows through qrels.
        ev = full_ev.groupby("aircraft_model", group_keys=False, sort=False).head(64).reset_index(drop=True)
        questions = [next(q for q in questions if q["aircraft_model"] == aircraft) for aircraft in ev.aircraft_model.unique()]
    elif args.limit:
        questions = questions[:args.limit]
    # The primary neural retrieval file remains GPU fp32 for the requested run.
    # CPU smoke/FP16 runs have distinct files and never mix timing distributions.
    stage_name = "neural_smoke" if args.smoke else "neural_retrieval"
    variant_suffix = "" if args.device in ("cuda", "directml") and args.precision == "fp32" and not args.smoke else f"_{args.device}_{args.precision}"
    dest = OUT / "results" / f"{stage_name}{variant_suffix}_runs.jsonl"
    completed = set()
    if dest.exists():
        completed = {(r["question_id"], r["method"]) for r in map(json.loads, dest.read_text(encoding="utf-8").splitlines())}
    pending = [q for q in questions if not all((q["question_id"], method) in completed for method in METHODS)]
    print(json.dumps({"phase": "start", "n_corpus": len(ev), "n_queries": len(questions), "n_pending": len(pending), "threads": args.threads, "smoke": args.smoke, "device": args.device, "precision": args.precision}), flush=True)
    if not pending:
        return
    start_time = time.perf_counter()
    if args.device == "directml":
        from directml_neural import LocalDirectML
        neural = LocalDirectML(args)
    else:
        neural = LocalNeural(args)
    embedding, cache_info = corpus_embeddings(ev, neural, args)
    v21 = _load_v21_functions(ROOT / "_pipeline_v21" / "p0_build_eval_v21.py")
    bm25 = BM25(ev, v21.tokenize)
    gran = GranularRetriever()
    global_ids = ev.evidence_id.to_numpy()
    global_index = {eid: i for i, eid in enumerate(global_ids)}
    reg = registry(args)
    prefix = f"{stage_name}{variant_suffix}"
    for index, q in enumerate(pending):
        # Online boundary: no question-id, label, split, track or answer is passed.
        query = str(q["question"])
        aircraft = str(q["aircraft_model"])
        eligible = ev.aircraft_model.eq(aircraft).to_numpy()
        q_started = time.perf_counter()
        dense = np.asarray(embedding @ neural.encode([query])[0])
        dense_pool = top(dense, eligible, args.candidate_pool)
        lexical = bm25.score(query)
        lexical_pool = top(lexical, eligible, args.candidate_pool)
        fused = rank_values(dense, eligible, args.candidate_pool) + rank_values(lexical, eligible, args.candidate_pool)
        r, h, w, _ = gran.scores(query, aircraft)
        multi = rank_values(dense, eligible, args.candidate_pool)
        for name, ranks in r.items():
            view_indices = gran.mod.top_indices(ranks, gran._eligible, args.candidate_pool)
            for rank, detail_index in enumerate(view_indices, 1):
                eid = gran.detail_ids[detail_index]
                if eid in global_index:
                    multi[global_index[eid]] += float(w[name]) / (60 + rank)
        multi_pool = top(multi, eligible, args.candidate_pool)
        ce_union = np.asarray(sorted(set(dense_pool[:args.matched_ce_pool]) | set(multi_pool[:args.matched_ce_pool])), dtype=int)
        ce_started = time.perf_counter()
        ce_values = neural.rerank(query, ev.iloc[ce_union].text.map(norm).tolist())
        ce_elapsed = time.perf_counter() - ce_started
        ce_map = dict(zip(ce_union.tolist(), ce_values.tolist()))
        def ce_sort(pool: np.ndarray) -> np.ndarray:
            return np.asarray(sorted(pool.tolist(), key=lambda i: (-ce_map[i], i)), dtype=int)
        selected = {
            METHODS[0]: dense_pool[:args.k],
            METHODS[1]: ce_sort(dense_pool[:args.protocol_ce_pool])[:args.k],
            METHODS[2]: top(fused, eligible, args.k),
            METHODS[3]: ce_sort(dense_pool[:args.matched_ce_pool])[:args.k],
            METHODS[4]: ce_sort(multi_pool[:args.matched_ce_pool])[:args.k],
        }
        pools = {METHODS[0]: dense_pool, METHODS[1]: dense_pool[:30], METHODS[2]: np.unique(np.concatenate((dense_pool, lexical_pool))), METHODS[3]: dense_pool[:50], METHODS[4]: multi_pool[:50]}
        with dest.open("a", encoding="utf-8") as handle:
            for method in METHODS:
                if (q["question_id"], method) in completed:
                    continue
                evidence_ids = global_ids[selected[method]].tolist()
                assert all(evmap[eid]["aircraft_model"] == aircraft for eid in evidence_ids)
                row = {
                    "question_id": q["question_id"], "question": query, "aircraft_model": aircraft,
                    "split": q["split"], "cluster_id": q["cluster_id"], "source_group": q["source_group"],
                    "method": method, "top_evidence_ids": ";".join(evidence_ids),
                    "candidate_evidence_ids": ";".join(global_ids[pools[method]].tolist()),
                    "candidate_pool_size": len(pools[method]),
                    "ce_pool_size": len(pools[method]) if "CE@" in method else 0,
                    "ce_scored_unique_pairs_this_query": len(ce_union),
                    "ce_unique_pair_runtime_s": ce_elapsed,
                    "grouped_neural_runtime_s": time.perf_counter() - q_started,
                    "synthetic_query": True, "gold_in_online_features": False,
                    "real_pretrained_neural_weights": True, "smoke_subsampled_corpus": args.smoke,
                    "neural_device": args.device, "neural_precision": args.precision,
                    **evaluate(q, evidence_ids, evmap),
                    **{f"entropy_{name}": value for name, value in h.items()},
                    **{f"weight_{name}": value for name, value in w.items()},
                }
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=True) + "\n")
                completed.add((q["question_id"], method))
        print(json.dumps({"phase": "queries", "completed": index + 1, "pending": len(pending), "ce_pairs": len(ce_union), "query_s": round(time.perf_counter() - q_started, 3), "elapsed_s": round(time.perf_counter() - start_time, 2)}, ensure_ascii=False), flush=True)
    frame = pd.DataFrame(map(json.loads, dest.read_text(encoding="utf-8").splitlines()))
    metrics = ["accepted_hit_at_10", "exact_hit_at_10", "exact_set_recall_at_10", "accepted_set_recall_at_10", "required_quote_support_fraction", "all_quote_support", "page_hit_at_10", "role_match_fraction"]
    summary = frame.groupby(["split", "method"])[metrics].agg(["mean", "count"]).reset_index()
    summary.columns = ["_".join(col).rstrip("_") if isinstance(col, tuple) else col for col in summary.columns]
    summary.to_csv(OUT / "results" / f"{prefix}_summary_split.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "n_questions_selected": len(questions), "n_completed_rows": len(frame), "n_methods": len(METHODS),
        "torch_version": torch.__version__, "torch_cuda_version": torch.version.cuda,
        "device": args.device, "precision": args.precision, "threads": args.threads,
        "device_name": torch.cuda.get_device_name(0) if args.device == "cuda" else getattr(neural, "device_name", "cpu"),
        "source_sha256": source_hash_at_start, "question_file_sha256": question_hash_at_start,
        "evaluated_question_ids": [q["question_id"] for q in questions],
        "dense_model_path": str(args.dense_path), "dense_weights_sha256": sha_file(args.dense_path / "model.safetensors"),
        "ce_model_path": str(args.ce_path), "ce_weights_sha256": sha_file(args.ce_path / "model.safetensors"),
        "dense_max_length": args.dense_max_length, "ce_max_length": args.ce_max_length,
        "embedding_cache": cache_info, "method_protocols": reg,
        "elapsed_s": time.perf_counter() - start_time, "qrels_only_posthoc": True,
        "smoke": args.smoke, "official_reproduction": False,
        "claim_boundary": "Real pretrained dense/CE baselines with documented candidate pools; IntentRAG uses original entropy weighted multi-view routing plus dense RRF. No source-specific retraining; no invented neural approximation.",
    }
    if args.device == "directml":
        manifest["directml_execution_profiles"] = neural.close_profiles()
    (OUT / "reports" / f"{prefix}_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
