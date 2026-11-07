"""FP32 DirectML backend exported from the existing local HF model weights."""
from __future__ import annotations

import ctypes
import gc
import json
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer


def dxgi_adapters() -> list[dict[str, Any]]:
    class GUID(ctypes.Structure):
        _fields_ = [("data", ctypes.c_ubyte * 16)]
    class LUID(ctypes.Structure):
        _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_int32)]
    class Desc(ctypes.Structure):
        _fields_ = [("description", ctypes.c_wchar * 128), ("vendor", ctypes.c_uint32),
            ("device", ctypes.c_uint32), ("subsys", ctypes.c_uint32), ("revision", ctypes.c_uint32),
            ("dedicated_video", ctypes.c_size_t), ("dedicated_system", ctypes.c_size_t),
            ("shared_system", ctypes.c_size_t), ("luid", LUID), ("flags", ctypes.c_uint32)]
    iid = GUID((ctypes.c_ubyte * 16).from_buffer_copy(uuid.UUID("770aae78-f26f-4dba-a829-253c83d1b387").bytes_le))
    factory = ctypes.c_void_p()
    dll = ctypes.WinDLL("dxgi")
    create = dll.CreateDXGIFactory1
    create.argtypes = [ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p)]
    create.restype = ctypes.c_long
    if create(ctypes.byref(iid), ctypes.byref(factory)) != 0:
        raise RuntimeError("Cannot enumerate DXGI GPU adapters")
    def method(pointer: ctypes.c_void_p, index: int, restype: Any, *argtypes: Any) -> Any:
        table = ctypes.cast(pointer, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        return ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(table[index])
    enum = method(factory, 12, ctypes.c_long, ctypes.c_uint32, ctypes.POINTER(ctypes.c_void_p))
    found = []
    for index in range(16):
        adapter = ctypes.c_void_p()
        if enum(factory, index, ctypes.byref(adapter)) != 0:
            break
        desc = Desc()
        get = method(adapter, 10, ctypes.c_long, ctypes.POINTER(Desc))
        if get(adapter, ctypes.byref(desc)) == 0:
            found.append({"device_id": index, "description": desc.description,
                "vendor_id": desc.vendor, "dedicated_video_bytes": desc.dedicated_video,
                "software_adapter": bool(desc.flags & 2)})
        method(adapter, 2, ctypes.c_ulong)(adapter)
    method(factory, 2, ctypes.c_ulong)(factory)
    return found


class DenseWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Precompute the additive 4D padding mask. This avoids tracing the HF
        # mask factory's vmap path into constants tied to the sample batch.
        additive = (1.0 - attention_mask[:, None, None, :].float()) * torch.finfo(torch.float32).min
        hidden = self.model(input_ids=input_ids, attention_mask=additive).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        mean = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        return torch.nn.functional.normalize(mean, p=2, dim=1)


class CEWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        additive = (1.0 - attention_mask[:, None, None, :].float()) * torch.finfo(torch.float32).min
        return self.model(input_ids=input_ids, attention_mask=additive).logits[:, 0]


class LocalDirectML:
    def __init__(self, args: Any):
        self.args = args
        if "DmlExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("DirectML requested but provider unavailable; refusing CPU fallback.")
        adapters = dxgi_adapters()
        selected = next((item for item in adapters if item["vendor_id"] == 0x10DE and not item["software_adapter"]), None)
        if selected is None:
            raise RuntimeError(f"NVIDIA DXGI adapter not found: {adapters}")
        self.device_name = selected["description"]
        self.dense_tokenizer = AutoTokenizer.from_pretrained(args.dense_path, local_files_only=True)
        self.ce_tokenizer = AutoTokenizer.from_pretrained(args.ce_path, local_files_only=True)
        self.cache = Path(__file__).resolve().parents[1] / "models" / "neural_onnx_fp32"
        self.cache.mkdir(parents=True, exist_ok=True)
        # The outer wrapper must itself be eval(), otherwise ONNX exporter
        # restores wrapper.training=True and recursively re-enables dropout in
        # the reference model after export, invalidating numerical comparison.
        dense_ref = DenseWrapper(AutoModel.from_pretrained(args.dense_path, local_files_only=True, attn_implementation="eager").eval()).eval()
        ce_ref = CEWrapper(AutoModelForSequenceClassification.from_pretrained(args.ce_path, local_files_only=True, attn_implementation="eager").eval()).eval()
        dense_path = self.cache / "dense_mean_normalized_mask4d.onnx"
        ce_path = self.cache / "mmarco_logits_mask4d.onnx"
        self.export(dense_ref, self.dense_tokenizer, dense_path, args.dense_max_length, False)
        self.export(ce_ref, self.ce_tokenizer, ce_path, args.ce_max_length, True)
        def session(path: Path) -> ort.InferenceSession:
            options = ort.SessionOptions()
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.enable_mem_pattern = False
            options.intra_op_num_threads = args.threads
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            options.enable_profiling = True
            options.profile_file_prefix = str(self.cache / path.stem)
            return ort.InferenceSession(str(path), sess_options=options,
                providers=[("DmlExecutionProvider", {"device_id": selected["device_id"]}), "CPUExecutionProvider"])
        self.dense_session = session(dense_path)
        self.ce_session = session(ce_path)
        dense_ref.eval()
        ce_ref.eval()
        validation = self.validate(dense_ref, ce_ref)
        validation.update({"onnx_version": onnx.__version__, "onnxruntime_version": ort.__version__,
            "providers": ort.get_available_providers(), "dxgi_adapters": adapters, "selected_adapter": selected,
            "dense_session_providers": self.dense_session.get_providers(), "ce_session_providers": self.ce_session.get_providers(),
            "fp32": True, "opset": 17, "source_weights_are_local": True})
        (self.cache / "directml_validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"phase": "directml_validation", **validation}, ensure_ascii=False), flush=True)
        del dense_ref, ce_ref
        gc.collect()
        if not validation["passed"]:
            raise RuntimeError("DirectML export validation failed; untrusted outputs are not ranked.")

    def input(self, tokenizer: Any, texts: list[str], max_length: int, pairs: list[str] | None = None) -> dict[str, np.ndarray]:
        encoded = tokenizer(texts, pairs, padding="max_length", truncation=True, max_length=max_length,
            return_tensors="np", return_token_type_ids=False)
        return {key: np.asarray(encoded[key], dtype=np.int64) for key in ("input_ids", "attention_mask")}

    def export(self, model: Any, tokenizer: Any, path: Path, max_length: int, pair: bool) -> None:
        if path.exists():
            return
        inputs = self.input(tokenizer, ["The evidence identifies a valve.", "Maintenance action requires an applicable procedure."], max_length,
            ["Description alone does not authorize replacement.", "Use the current aircraft manual."] if pair else None)
        with torch.inference_mode():
            torch.onnx.export(model, tuple(torch.from_numpy(inputs[key]) for key in ("input_ids", "attention_mask")), str(path),
                input_names=["input_ids", "attention_mask"], output_names=["output"],
                dynamic_axes={"input_ids": {0: "batch"}, "attention_mask": {0: "batch"}, "output": {0: "batch"}},
                opset_version=17, dynamo=False, do_constant_folding=True)
        onnx.checker.check_model(str(path))

    def validate(self, dense_ref: Any, ce_ref: Any) -> dict[str, Any]:
        texts = ["A valve controls airflow.", "The manual describes a warning indication.", "An approved applicable procedure is required before replacement."]
        query = "What evidence authorizes valve replacement?"
        dense_inputs = self.input(self.dense_tokenizer, texts, self.args.dense_max_length)
        ce_inputs = self.input(self.ce_tokenizer, [query] * len(texts), self.args.ce_max_length, texts)
        with torch.inference_mode():
            expected_dense = dense_ref(**{key: torch.from_numpy(value) for key, value in dense_inputs.items()}).numpy()
            expected_ce = ce_ref(**{key: torch.from_numpy(value) for key, value in ce_inputs.items()}).numpy()
            source_dense_inputs = {key: torch.from_numpy(value) for key, value in dense_inputs.items()}
            source_hidden = dense_ref.model(**source_dense_inputs).last_hidden_state
            source_mask = source_dense_inputs["attention_mask"].unsqueeze(-1).float()
            source_dense = torch.nn.functional.normalize((source_hidden * source_mask).sum(1) / source_mask.sum(1).clamp(min=1), dim=1).numpy()
            source_ce = ce_ref.model(**{key: torch.from_numpy(value) for key, value in ce_inputs.items()}).logits[:, 0].numpy()
        actual_dense = self.dense_session.run(None, dense_inputs)[0]
        actual_ce = self.ce_session.run(None, ce_inputs)[0]
        cpu_dense_session = ort.InferenceSession(str(self.cache / "dense_mean_normalized_mask4d.onnx"), providers=["CPUExecutionProvider"])
        cpu_ce_session = ort.InferenceSession(str(self.cache / "mmarco_logits_mask4d.onnx"), providers=["CPUExecutionProvider"])
        cpu_dense = cpu_dense_session.run(None, dense_inputs)[0]
        cpu_ce = cpu_ce_session.run(None, ce_inputs)[0]
        dense_error = float(np.max(np.abs(actual_dense - expected_dense)))
        ce_error = float(np.max(np.abs(actual_ce - expected_ce)))
        dense_rank = bool(np.array_equal(np.argsort(-actual_dense @ actual_dense[0]), np.argsort(-expected_dense @ expected_dense[0])))
        ce_rank = bool(np.array_equal(np.argsort(-actual_ce), np.argsort(-expected_ce)))
        extra_cases = []
        for case_name, case_texts in [("batch1_short", ["The valve is described."]),
                ("batch2_mixed_full_length", ["The evidence identifies a valve. " * 150, "仅凭描述性证据不能授权更换部件。"] )]:
            d = self.input(self.dense_tokenizer, case_texts, self.args.dense_max_length)
            c = self.input(self.ce_tokenizer, [query] * len(case_texts), self.args.ce_max_length, case_texts)
            with torch.inference_mode():
                td = dense_ref(**{key: torch.from_numpy(value) for key, value in d.items()}).numpy()
                tc = ce_ref(**{key: torch.from_numpy(value) for key, value in c.items()}).numpy()
            extra_cases.append({"case": case_name, "batch": len(case_texts),
                "dense_dml_error": float(np.max(np.abs(self.dense_session.run(None, d)[0] - td))),
                "dense_onnx_cpu_error": float(np.max(np.abs(cpu_dense_session.run(None, d)[0] - td))),
                "ce_dml_error": float(np.max(np.abs(self.ce_session.run(None, c)[0] - tc))),
                "ce_onnx_cpu_error": float(np.max(np.abs(cpu_ce_session.run(None, c)[0] - tc)))})
        source_error_dense = float(np.max(np.abs(expected_dense - source_dense)))
        source_error_ce = float(np.max(np.abs(expected_ce - source_ce)))
        return {"dense_max_abs_error": dense_error, "ce_max_abs_error": ce_error,
            "onnx_cpu_dense_max_abs_error": float(np.max(np.abs(cpu_dense - expected_dense))),
            "onnx_cpu_ce_max_abs_error": float(np.max(np.abs(cpu_ce - expected_ce))),
            "dense_smoke_rank_identical": dense_rank, "ce_smoke_rank_identical": ce_rank,
            "source_2d_vs_wrapper4d_dense_error": source_error_dense,
            "source_2d_vs_wrapper4d_ce_error": source_error_ce,
            "additional_shape_cases": extra_cases,
            "passed": dense_error < 1e-3 and ce_error < 1e-3 and dense_rank and ce_rank and source_error_dense < 1e-3 and source_error_ce < 1e-3 and all(max(float(value) for key, value in case.items() if key.endswith("_error")) < 1e-3 for case in extra_cases),
            "validation_inputs_documented": texts, "validation_query": query}

    def encode(self, texts: list[str]) -> np.ndarray:
        inputs = self.input(self.dense_tokenizer, texts, self.args.dense_max_length)
        return self.dense_session.run(None, inputs)[0].astype(np.float32)

    def rerank(self, query: str, texts: list[str]) -> np.ndarray:
        chunks = []
        for start in range(0, len(texts), self.args.ce_batch_size):
            batch = texts[start:start + self.args.ce_batch_size]
            inputs = self.input(self.ce_tokenizer, [query] * len(batch), self.args.ce_max_length, batch)
            chunks.append(self.ce_session.run(None, inputs)[0])
        return np.concatenate(chunks).astype(np.float32) if chunks else np.asarray([], dtype=np.float32)

    def close_profiles(self) -> dict[str, Any]:
        paths = [self.dense_session.end_profiling(), self.ce_session.end_profiling()]
        provider_counts: dict[str, int] = {}
        for path in paths:
            for event in json.loads(Path(path).read_text(encoding="utf-8")):
                provider = event.get("args", {}).get("provider")
                if provider:
                    provider_counts[provider] = provider_counts.get(provider, 0) + 1
        return {"profile_paths": paths, "executed_node_counts_by_provider": provider_counts}
