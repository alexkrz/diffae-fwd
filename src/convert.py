#!/usr/bin/env python3
"""Analyze PyTorch checkpoints and convert them to safetensors.

Examples:
  python convert.py --input checkpoints
  python convert.py --input checkpoints/ffhq256_autoenc/last.ckpt --output-dir checkpoints_safetensors
  python convert.py --input checkpoints --analyze-only
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import save_file

CHECKPOINT_EXTENSIONS = {".ckpt", ".pt", ".pth", ".bin", ".pkl"}


def _tensor_nbytes(t: torch.Tensor) -> int:
    return int(t.numel() * t.element_size())


def _format_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(n)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{n} B"


def _accumulate_tensor_bytes(obj: Any) -> int:
    if torch.is_tensor(obj):
        return _tensor_nbytes(obj)
    if isinstance(obj, dict):
        return sum(_accumulate_tensor_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_accumulate_tensor_bytes(v) for v in obj)
    return 0


def _payload_tensor_breakdown(payload: Any) -> dict[str, int]:
    if isinstance(payload, dict):
        breakdown = {k: _accumulate_tensor_bytes(v) for k, v in payload.items()}
    else:
        breakdown = {"<root>": _accumulate_tensor_bytes(payload)}
    breakdown = {k: v for k, v in breakdown.items() if v > 0}
    return dict(sorted(breakdown.items(), key=lambda kv: kv[1], reverse=True))


def _is_tensor_dict(obj: Any) -> bool:
    return isinstance(obj, dict) and all(isinstance(k, str) and torch.is_tensor(v) for k, v in obj.items())


def _find_tensor_dict(payload: Any) -> tuple[str, dict[str, torch.Tensor]]:
    if _is_tensor_dict(payload):
        return "<root>", payload

    if isinstance(payload, dict):
        if "state_dict" in payload and _is_tensor_dict(payload["state_dict"]):
            return "state_dict", payload["state_dict"]
        for key in ("model", "model_state_dict", "ema", "ema_state_dict"):
            if key in payload and _is_tensor_dict(payload[key]):
                return key, payload[key]

    raise ValueError("Could not find a tensor state dict in checkpoint")


def _to_supported_tensor(value: Any) -> torch.Tensor | None:
    if torch.is_tensor(value):
        return value
    if isinstance(value, np.ndarray) and value.dtype.kind in {"b", "i", "u", "f"}:
        return torch.from_numpy(value)
    return None


def _extract_tensors_from_mapping(mapping: dict[str, Any], prefix: str = "") -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in mapping.items():
        if not isinstance(key, str):
            continue
        full_key = f"{prefix}{key}"
        tensor = _to_supported_tensor(value)
        if tensor is not None:
            out[full_key] = tensor
            continue
        if isinstance(value, dict):
            out.update(_extract_tensors_from_mapping(value, prefix=f"{full_key}."))
    return out


def _find_latent_tensor_dict(payload: Any) -> tuple[str, dict[str, torch.Tensor]]:
    if isinstance(payload, dict):
        tensors = _extract_tensors_from_mapping(payload)
        if tensors:
            return "latent", tensors
    raise ValueError("Could not find latent tensors in payload")


def _extract_ema_state_dict(
    state_key: str, state_dict: dict[str, torch.Tensor]
) -> tuple[str, dict[str, torch.Tensor]]:
    """Return EMA-only weights with optional prefix stripped.

    Supported input forms:
    - full lightning state_dict with keys like "ema_model.encoder..."
    - checkpoint key already named "ema" or "ema_state_dict"
    - plain EMA dict with already-stripped keys
    """
    if state_key in {"ema", "ema_state_dict"}:
        if any(k.startswith("ema_model.") for k in state_dict):
            out = {k[len("ema_model.") :]: v for k, v in state_dict.items() if k.startswith("ema_model.")}
            if out:
                return "ema_model_stripped", out
        return state_key, state_dict

    prefixed = {k[len("ema_model.") :]: v for k, v in state_dict.items() if k.startswith("ema_model.")}
    if prefixed:
        return "ema_model_stripped", prefixed

    # Fallback: if this already looks like a plain model dict, use it as-is.
    # This allows using --ema-only with already-converted EMA safetensors/pt files.
    looks_plain = any(
        k.startswith("encoder.") or k.startswith("time_embed.") or k.startswith("input_blocks.") for k in state_dict
    )
    if looks_plain:
        return "<plain_model>", state_dict

    raise ValueError("Could not find EMA weights (expected keys prefixed with 'ema_model.')")


def _tensor_numel(t: torch.Tensor) -> int:
    return int(t.numel())


def _analyze_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
    dtypes = Counter()
    devices = Counter()
    prefixes = Counter()
    total_tensors = 0
    total_params = 0
    total_tensor_bytes = 0

    for key, tensor in state_dict.items():
        total_tensors += 1
        total_params += _tensor_numel(tensor)
        total_tensor_bytes += _tensor_nbytes(tensor)
        dtypes[str(tensor.dtype)] += 1
        devices[str(tensor.device)] += 1
        prefix = key.split(".", 1)[0]
        prefixes[prefix] += 1

    largest = sorted(state_dict.items(), key=lambda kv: kv[1].numel(), reverse=True)[:5]
    largest_tensors = [
        {
            "name": name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "numel": int(tensor.numel()),
        }
        for name, tensor in largest
    ]

    return {
        "tensor_count": total_tensors,
        "parameter_count": total_params,
        "tensor_bytes": total_tensor_bytes,
        "dtypes": dict(dtypes),
        "devices": dict(devices),
        "top_level_prefix_counts": dict(prefixes),
        "largest_tensors": largest_tensors,
    }


def _discover_checkpoint_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    files = [p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() in CHECKPOINT_EXTENSIONS]
    return sorted(files)


def _to_cpu_contiguous(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        tensor = value.detach().cpu()
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        out[key] = tensor
    return out


def _build_output_path(input_file: Path, input_root: Path, output_root: Path) -> Path:
    rel = input_file.relative_to(input_root) if input_root.is_dir() else Path(input_file.name)
    return (output_root / rel).with_suffix(".safetensors")


def _print_report(file_path: Path, state_key: str, analysis: dict[str, Any], payload: Any) -> None:
    file_size = file_path.stat().st_size
    state_tensor_bytes = int(analysis["tensor_bytes"])
    payload_breakdown = _payload_tensor_breakdown(payload)
    payload_tensor_total = sum(payload_breakdown.values())

    print(f"\n=== {file_path} ===")
    print(f"state key: {state_key}")
    print(f"file size: {_format_bytes(file_size)}")
    print(
        "selected state_dict tensor bytes:",
        f"{_format_bytes(state_tensor_bytes)} ({state_tensor_bytes / max(file_size, 1):.1%} of file)",
    )
    print(
        "all checkpoint tensor bytes:",
        f"{_format_bytes(payload_tensor_total)} ({payload_tensor_total / max(file_size, 1):.1%} of file)",
    )
    if payload_breakdown:
        print("tensor-byte breakdown by top-level key:")
        for key, nbytes in list(payload_breakdown.items())[:10]:
            print(f"  - {key}: {_format_bytes(nbytes)}")

    if isinstance(payload, dict):
        print("top-level keys:", ", ".join(sorted(payload.keys())))
        for key in ("epoch", "global_step"):
            if key in payload:
                print(f"{key}: {payload[key]}")
    print(f"tensor count: {analysis['tensor_count']}")
    print(f"parameter count: {analysis['parameter_count']:,}")
    print("dtypes:", analysis["dtypes"])
    print("devices:", analysis["devices"])
    print("largest tensors:")
    for item in analysis["largest_tensors"]:
        print(f"  - {item['name']}: shape={item['shape']}, dtype={item['dtype']}, numel={item['numel']:,}")


def convert_one(
    file_path: Path, input_root: Path, output_root: Path, analyze_only: bool, ema_only: bool
) -> dict[str, Any]:
    payload = torch.load(file_path, map_location="cpu", weights_only=False)
    try:
        state_key, state_dict = _find_tensor_dict(payload)
    except ValueError:
        if file_path.suffix.lower() == ".pkl":
            state_key, state_dict = _find_latent_tensor_dict(payload)
        else:
            raise

    export_state_key = state_key
    export_state = state_dict
    if ema_only:
        if export_state_key == "latent":
            raise ValueError("--ema-only is not compatible with latent .pkl exports")
        export_state_key, export_state = _extract_ema_state_dict(state_key, state_dict)

    analysis = _analyze_state_dict(export_state)
    _print_report(file_path, export_state_key, analysis, payload)

    result = {
        "file": str(file_path),
        "state_key": export_state_key,
        "ema_only": ema_only,
        "analysis": analysis,
        "converted": False,
        "output": None,
    }

    if analyze_only:
        return result

    output_path = _build_output_path(file_path, input_root, output_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "source": str(file_path),
        "state_key": export_state_key,
        "ema_only": str(ema_only),
        "tensor_count": str(analysis["tensor_count"]),
        "parameter_count": str(analysis["parameter_count"]),
    }
    cpu_state = _to_cpu_contiguous(export_state)
    save_file(cpu_state, str(output_path), metadata=metadata)

    print(f"saved: {output_path}")
    result["converted"] = True
    result["output"] = str(output_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze and convert checkpoints to safetensors")
    parser.add_argument("--input", required=True, type=Path, help="Checkpoint file or directory")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for safetensors output (default: same directory tree under input parent)",
    )
    parser.add_argument("--analyze-only", action="store_true", help="Only print analysis, do not write safetensors")
    parser.add_argument(
        "--ema-only",
        action="store_true",
        help="Export only EMA model weights (extracts and strips 'ema_model.' keys when present)",
    )
    parser.add_argument("--json-report", type=Path, default=None, help="Optional path to save JSON analysis report")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path: Path = args.input.resolve()
    files = _discover_checkpoint_files(input_path)
    if not files:
        raise FileNotFoundError(f"No checkpoint files found under: {input_path}")

    output_root = args.output_dir.resolve() if args.output_dir else (input_path.parent / "safetensors")

    print(f"found {len(files)} checkpoint file(s)")
    if args.ema_only:
        print("mode: EMA-only export")
    if not args.analyze_only:
        print(f"output dir: {output_root}")

    results = []
    failures = 0
    for file_path in files:
        try:
            results.append(convert_one(file_path, input_path, output_root, args.analyze_only, args.ema_only))
        except Exception as exc:
            failures += 1
            print(f"\n=== {file_path} ===")
            print(f"error: {type(exc).__name__}: {exc}")

    if args.json_report:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nreport saved: {args.json_report}")

    print(f"\ncomplete: {len(results)} succeeded, {failures} failed")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
