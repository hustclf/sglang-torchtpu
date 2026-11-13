# SPDX-License-Identifier: Apache-2.0
"""Write each direct_compile specialization as StableHLO for offline analysis.

Enabled by ``SGLANG_TPU_EXPORT_STABLEHLO_DIR``. The file set matches
tpu-infer-loop ``tools/torchtpu-export`` so hlo-roofline reads it unchanged:
``<name>.stablehlo.mlir`` (no locations), ``.stablehlo.loc.mlir``,
``.stablehlo.mlirbc`` and ``<name>.manifest.json``, whose ``inputs`` list is
in ``@main`` argument order.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import torch
from torch_tpu._internal.compile import tpu_torch_compile

from sglang_tpu._logging import get_logger
from sglang_tpu.envs import envs

logger = get_logger(__name__)

_exported_count = 0


def export_dir() -> Path | None:
    value = envs.SGLANG_TPU_EXPORT_STABLEHLO_DIR.get()
    return Path(value) if value else None


def input_label(*, kind: str, name: str, tensor: torch.Tensor) -> dict:
    return {
        "kind": kind,
        "name": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
    }


def export_specialization(
    *,
    out_dir: Path,
    graph_name: str,
    specialization: str,
    result_tensors: list[torch.Tensor],
    graph_inputs: list[torch.Tensor],
    input_labels: list[dict],
    output_counts: dict[str, int],
) -> None:
    """Lower one traced specialization to StableHLO and write it to ``out_dir``."""
    global _exported_count
    if len(input_labels) != len(graph_inputs):
        raise RuntimeError(
            f"StableHLO export: {len(input_labels)} labels for "
            f"{len(graph_inputs)} graph inputs"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = _file_stem(graph_name=graph_name, specialization=specialization)
    module = tpu_torch_compile.build_mlir(result_tensors, graph_inputs)
    text = tpu_torch_compile.serialize_mlir_text(module, enable_debug_info=False)
    files = {
        f"{stem}.stablehlo.mlir": text.encode(),
        f"{stem}.stablehlo.loc.mlir": tpu_torch_compile.serialize_mlir_text(
            module, enable_debug_info=True
        ).encode(),
        f"{stem}.stablehlo.mlirbc": tpu_torch_compile.serialize_mlir_portable_artifact(
            module
        ),
    }
    for file_name, data in files.items():
        (out_dir / file_name).write_bytes(data)
    main_args = _main_argument_count(text)
    manifest = {
        "graph": graph_name,
        "specialization": specialization,
        "inputs": input_labels,
        "outputs": output_counts,
        "stablehlo_ops": text.count("stablehlo."),
        "custom_calls": text.count("custom_call"),
        "main_arguments": main_args,
        # Complete only; semantic equivalence with execution is not checked here.
        "ok": main_args == len(input_labels),
        "sha256": {
            name: hashlib.sha256(data).hexdigest() for name, data in files.items()
        },
    }
    (out_dir / f"{stem}.manifest.json").write_text(
        json.dumps(manifest, indent=1) + "\n"
    )
    _exported_count += 1
    logger.info(
        "[stablehlo export] wrote %s (%d inputs, %d custom_calls, ok=%s)",
        out_dir / f"{stem}.stablehlo.mlir",
        len(input_labels),
        manifest["custom_calls"],
        manifest["ok"],
    )
    exit_after = envs.SGLANG_TPU_EXPORT_STABLEHLO_EXIT_AFTER.get()
    if exit_after > 0 and _exported_count >= exit_after:
        # Same exit as SPMD verification: stop before warmup executes the graphs.
        logger.info("[stablehlo export] exported %d graphs, exiting", _exported_count)
        raise SystemExit(0)


def _file_stem(*, graph_name: str, specialization: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{graph_name}-{specialization}")[:120]
    short_hash = hashlib.sha256(specialization.encode()).hexdigest()[:12]
    return f"{readable}-{short_hash}"


def _main_argument_count(text: str) -> int:
    match = re.search(r"func\.func\s+(?:public\s+)?@main\(([^)]*)\)", text)
    if match is None or not match.group(1).strip():
        return 0
    return match.group(1).count("%arg")
