"""Measure native llama.cpp slot KV reuse on local GGUF prompts.

Owns one hidden, loopback-only server process and terminates only that process.
JSONL rows accept prompt, optional seed_prompt, method/query_id/namespace.
Cold target -> (optional distinct seed) -> warm target uses slot 0 throughout.
All raw responses, server logs, tokenization counts, and timing counters survive.
This measures actual engine context reuse, not token-overlap eligibility.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_SERVER = Path(r"${LLAMA_SERVER}")
DEFAULT_BACKEND = DEFAULT_SERVER.parent / "cuda_v13" / "ggml-cuda.dll"
DEFAULT_MODEL = Path(r"${MODEL_PATH}")
EXPERIMENT = Path(__file__).resolve().parents[1]


def request_json(base: str, endpoint: str, body: dict[str, Any] | None = None, timeout: int = 180) -> Any:
    request = urllib.request.Request(
        base + endpoint,
        data=None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="GET" if body is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class OwnedServer:
    def __init__(self, args: argparse.Namespace, output: Path) -> None:
        self.args = args
        self.output = output
        self.process: subprocess.Popen[bytes] | None = None
        self.log_handle: Any = None
        self.base = f"http://127.0.0.1:{args.port}"
        self.command: list[str] = []

    def __enter__(self) -> "OwnedServer":
        for path in (self.args.server_exe, self.args.model_path, self.args.backend_dll):
            if not path.is_file():
                raise FileNotFoundError(path)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            if probe.connect_ex(("127.0.0.1", self.args.port)) == 0:
                raise RuntimeError(f"Port {self.args.port} already occupied; refusing to adopt or replace its process.")
        env = os.environ.copy()
        # This bundled build expects a complete backend DLL path, not its directory.
        env["GGML_BACKEND_PATH"] = str(self.args.backend_dll)
        env["PATH"] = str(self.args.backend_dll.parent) + os.pathsep + env.get("PATH", "")
        self.command = [
            str(self.args.server_exe), "-m", str(self.args.model_path),
            "--host", "127.0.0.1", "--port", str(self.args.port),
            "--parallel", "1", "--ctx-size", str(self.args.ctx_size),
            "--gpu-layers", "all", "--device", "CUDA0", "--offline",
            # Keep only the current fixed slot. A global RAM prompt cache could
            # restore the prior cold target instead of measuring seed -> target.
            "--cache-prompt", "--cache-ram", "0", "--no-cache-idle-slots", "--slots", "--metrics",
            "--no-context-shift", "--no-cont-batching", "--seed", str(self.args.seed),
            "--log-verbosity", "4",
        ]
        self.log_handle = (self.output / "llama_server.log").open("wb")
        self.process = subprocess.Popen(
            self.command, env=env, stdout=self.log_handle, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        write_json(self.output / "owned_server.json", {
            "pid": self.process.pid, "command": self.command,
            "backend_dll": str(self.args.backend_dll), "started_utc": datetime.now(timezone.utc).isoformat(),
        })
        deadline = time.monotonic() + self.args.startup_timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                self.__exit__(None, None, None)
                raise RuntimeError(f"Owned llama-server exited at startup; see {self.output / 'llama_server.log'}")
            try:
                if request_json(self.base, "/health", timeout=3).get("status") == "ok":
                    return self
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
                pass
            time.sleep(0.25)
        self.__exit__(None, None, None)
        raise TimeoutError("Owned llama-server did not become ready within startup timeout.")

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None


def tokenize(base: str, prompt: str) -> list[int]:
    response = request_json(base, "/tokenize", {"content": prompt, "add_special": False, "parse_special": True})
    return [int(token) for token in response["tokens"]]


def lcp(left: list[int], right: list[int]) -> int:
    count = 0
    for one, two in zip(left, right):
        if one != two:
            break
        count += 1
    return count


def completion(base: str, prompt: str, cache: bool, args: argparse.Namespace) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    response = request_json(base, "/completion", {
        "prompt": prompt, "id_slot": 0, "cache_prompt": cache,
        "n_predict": args.n_predict, "temperature": 0.0, "seed": args.seed,
        "top_k": 1, "stream": False, "return_tokens": True,
    }, timeout=args.request_timeout)
    return response, time.perf_counter() - started


def counter_record(response: dict[str, Any]) -> dict[str, Any]:
    timings = response.get("timings", {})
    record = {f"timing_{key}": value for key, value in timings.items() if isinstance(value, (int, float, bool))}
    # In bundled build 631109b34, top-level tokens_cached is final slot
    # occupancy (prompt + decoded tokens), NOT tokens reused at request start.
    if "tokens_cached" in response:
        record["engine_slot_tokens_final"] = response["tokens_cached"]
    record["actual_reused_prompt_tokens"] = timings.get("cache_n", "")
    for key in ("tokens_evaluated", "tokens_predicted", "id_slot", "truncated", "stopped_eos", "stopped_limit"):
        if key in response:
            record[key] = response[key]
    record["response_top_level_keys"] = ";".join(sorted(response))
    return record


def smoke_prompts() -> list[dict[str, str]]:
    lines = [
        f"Source record {i:03d}: The document identifies a valve and an associated indication. "
        "A description establishes component function; it does not independently authorize replacement."
        for i in range(20)
    ]
    prefix = "signature: Aircraft=Boeing737;revision=R1;authority=description-only\n" + "\n".join(lines)
    suffix = "\nQuestion: Is valve replacement authorized by this evidence?\nAnswer:"
    return [
        {"test_id": "cold_full", "prompt": prefix + suffix, "cache": False},
        {"test_id": "warm_exact", "prompt": prefix + suffix, "cache": True},
        {"test_id": "warm_query_change", "prompt": prefix + "\nQuestion: What information would authorize replacement?\nAnswer:", "cache": True},
        {"test_id": "warm_signature_change", "prompt": (prefix + suffix).replace("Aircraft=Boeing737", "Aircraft=Cessna172"), "cache": True},
    ]


def run_smoke(base: str, args: argparse.Namespace, output: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []
    previous_tokens: list[int] = []
    for item in smoke_prompts():
        prompt = str(item["prompt"])
        tokens = tokenize(base, prompt)
        response, latency = completion(base, prompt, bool(item["cache"]), args)
        row = {
            "test_id": item["test_id"], "cache_prompt": item["cache"],
            "prompt_token_count_native": len(tokens), "lcp_with_previous_prompt_native": lcp(tokens, previous_tokens),
            "latency_s": latency, "answer": response.get("content", ""), **counter_record(response),
        }
        rows.append(row)
        raw.append({"input": item, "response": response, "slots": request_json(base, "/slots")})
        previous_tokens = tokens
        print(json.dumps({key: value for key, value in row.items() if key != "answer"}, ensure_ascii=False), flush=True)
    exact_same = rows[0]["answer"] == rows[1]["answer"]
    cached = [int(row.get("timing_cache_n", -1)) for row in rows]
    conserved = all(
        int(row.get("timing_cache_n", -1)) + int(row.get("timing_prompt_n", -1)) == row["prompt_token_count_native"]
        for row in rows
    )
    report = {
        "request_count": len(rows), "actual_counter_present": all(value >= 0 for value in cached),
        "exact_cold_warm_output_identical": exact_same, "native_tokens_cached": cached,
        "prompt_token_accounting_conserved": conserved,
        "passed": len(rows) == 4 and conserved and cached[0] == 0 and cached[1] > 0 and cached[2] > 0 and 0 <= cached[3] < cached[2],
        "output_stability_status": "identical" if exact_same else "greedy_text_changed; inspect saved answers before claiming semantic stability",
        "startup_timing_excluded_from_claims": True,
        "counter_semantics": "timings.cache_n counts reused input tokens; timings.prompt_n counts newly evaluated input tokens; top-level tokens_cached is final slot occupancy.",
        "claim_boundary": "Native slot KV reuse and prefill timing; no maintenance correctness or safety validation.",
    }
    write_csv(output / "kv_smoke_requests.csv", rows)
    write_json(output / "kv_smoke_raw_responses.json", raw)
    write_json(output / "kv_smoke_report.json", report)
    return report


def load_rows(path: Path, max_rows: int) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    return rows[:max_rows] if max_rows else rows


def run_pairs(base: str, args: argparse.Namespace, output: Path) -> dict[str, Any]:
    rows = load_rows(args.prompts, args.max_rows)
    results: list[dict[str, Any]] = []
    requests = 0
    warmups = []
    if rows:
        first_prompt = str(rows[0].get("prompt", str(rows[0].get("prefix", "")) + str(rows[0].get("suffix", ""))))
        for _ in range(args.warmup_requests):
            response, latency = completion(base, first_prompt, False, args)
            warmups.append({"response": response, "latency_s": latency})
    write_json(output / "kv_unscored_warmup.json", warmups)
    with (output / "kv_pairs_raw_responses.jsonl").open("w", encoding="utf-8") as raw_handle:
        for index, row in enumerate(rows):
            prompt = str(row.get("prompt", str(row.get("prefix", "")) + str(row.get("suffix", ""))))
            seed_prompt = str(row.get("seed_prompt", prompt))
            target_tokens = tokenize(base, prompt)
            seed_tokens = tokenize(base, seed_prompt)
            cold, cold_latency = completion(base, prompt, False, args)
            requests += 1
            seed_response = None
            if seed_prompt != prompt:
                seed_response, _ = completion(base, seed_prompt, False, args)
                requests += 1
            actual_seed_response = seed_response if seed_response is not None else cold
            generated_ids = actual_seed_response.get("tokens", [])
            seed_context_tokens = seed_tokens + [int(value) for value in generated_ids[:-1] if isinstance(value, int)]
            seed_slot_snapshot = request_json(base, "/slots")
            warm, warm_latency = completion(base, prompt, True, args)
            requests += 1
            record: dict[str, Any] = {
                "row_index": index, "method": row.get("method", ""), "query_id": row.get("query_id", ""),
                "namespace": row.get("namespace", ""), "prompt_sha256": digest(prompt),
                "seed_prompt_sha256": digest(seed_prompt), "prompt_token_count_native": len(target_tokens),
                "seed_token_count_native": len(seed_tokens), "seed_target_lcp_native": lcp(seed_tokens, target_tokens),
                "seed_context_target_lcp_native": lcp(seed_context_tokens, target_tokens),
                "cold_latency_s": cold_latency, "warm_latency_s": warm_latency,
                "outputs_identical": cold.get("content") == warm.get("content"),
                "cold_answer": cold.get("content", ""), "warm_answer": warm.get("content", ""),
                **{f"cold_{key}": value for key, value in counter_record(cold).items()},
                **{f"warm_{key}": value for key, value in counter_record(warm).items()},
            }
            cold_ms = float(cold.get("timings", {}).get("prompt_ms", 0))
            warm_ms = float(warm.get("timings", {}).get("prompt_ms", 0))
            record["prefill_speedup_ratio"] = cold_ms / warm_ms if warm_ms > 0 else ""
            record["prefill_ms_saved"] = cold_ms - warm_ms
            record["cold_token_accounting_conserved"] = int(cold.get("timings", {}).get("cache_n", -1)) + int(cold.get("timings", {}).get("prompt_n", -1)) == len(target_tokens)
            record["warm_token_accounting_conserved"] = int(warm.get("timings", {}).get("cache_n", -1)) + int(warm.get("timings", {}).get("prompt_n", -1)) == len(target_tokens)
            record["cold_cache_zero"] = int(cold.get("timings", {}).get("cache_n", -1)) == 0
            record["warm_reuse_bounded_by_seed_lcp"] = 0 <= int(warm.get("timings", {}).get("cache_n", -1)) <= lcp(seed_tokens, target_tokens)
            record["warm_reuse_bounded_by_seed_context_lcp"] = 0 <= int(warm.get("timings", {}).get("cache_n", -1)) <= lcp(seed_context_tokens, target_tokens)
            record["warm_prompt_cache_fraction"] = int(warm.get("timings", {}).get("cache_n", 0)) / max(1, len(target_tokens))
            record["timing_measured_with_concurrent_ollama_qa"] = True
            results.append(record)
            raw_handle.write(json.dumps({"input": row, "cold": cold, "seed": seed_response, "seed_slot_before_warm": seed_slot_snapshot, "warm": warm}, ensure_ascii=False) + "\n")
            raw_handle.flush()
            write_csv(output / "kv_pairs.csv", results)
            if not record["cold_token_accounting_conserved"] or not record["warm_token_accounting_conserved"]:
                raise RuntimeError(f"Native input-token conservation failed at row {index}; raw responses retained.")
            print(json.dumps({"completed_rows": index + 1, "total_rows": len(rows), "method": record["method"], "warm_reused_prompt_tokens": warm.get("timings", {}).get("cache_n"), "outputs_identical": record["outputs_identical"]}, ensure_ascii=False), flush=True)
    report = {
        "rows": len(results), "request_count": requests,
        "unscored_warmup_requests": len(warmups),
        "actual_counter_present": all(isinstance(row.get("warm_actual_reused_prompt_tokens"), (int, float)) for row in results),
        "identical_output_count": sum(bool(row["outputs_identical"]) for row in results),
        "claim_boundary": "Actual native engine KV reuse for fixed-slot greedy decoding. Timing is paired and includes no retrieval; small numerical differences may change greedy text.",
        "server_protocol": "Cold target cache_prompt=false; optionally prime distinct seed false; warm target true; id_slot=0; identical settings for every method.",
        "counter_semantics": "timings.cache_n counts reused input tokens; timings.prompt_n newly evaluated input tokens. top-level tokens_cached is final slot occupancy.",
        "global_ram_prompt_cache_disabled": True,
        "all_token_accounting_conserved": all(row["cold_token_accounting_conserved"] and row["warm_token_accounting_conserved"] for row in results),
        "all_cold_cache_zero": all(row["cold_cache_zero"] for row in results),
        "all_warm_reuse_bounded_by_seed_lcp": all(row["warm_reuse_bounded_by_seed_lcp"] for row in results),
        "all_warm_reuse_bounded_by_seed_context_lcp": all(row["warm_reuse_bounded_by_seed_context_lcp"] for row in results),
        "concurrent_ollama_qa_on_11434": True,
        "timing_claim_boundary": "Concurrent local Qwen 3B QA uses GPU resources. Paired prefill timings are observed under contention and cannot be reported as isolated deployment throughput or pure cache speedup.",
        "revision_and_authority_validation": "Prompt markers are unverified; this experiment validates computational prefix reuse, not independent admissibility or authorization.",
    }
    write_json(output / "kv_pairs_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-exe", type=Path, default=DEFAULT_SERVER)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--backend-dll", type=Path, default=DEFAULT_BACKEND)
    parser.add_argument("--port", type=int, default=11435)
    parser.add_argument("--ctx-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20261014)
    parser.add_argument("--n-predict", type=int, default=32)
    parser.add_argument("--startup-timeout", type=int, default=60)
    parser.add_argument("--request-timeout", type=int, default=180)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.smoke and args.prompts is None:
        parser.error("Provide --smoke or --prompts JSONL.")
    output = args.output or EXPERIMENT / "results" / ("kv_smoke" if args.smoke else "real_kv")
    output.mkdir(parents=True, exist_ok=True)
    with OwnedServer(args, output) as server:
        report = run_smoke(server.base, args, output) if args.smoke else run_pairs(server.base, args, output)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.smoke and not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
