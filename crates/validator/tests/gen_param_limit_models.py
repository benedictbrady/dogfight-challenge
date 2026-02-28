#!/usr/bin/env python3
"""Generate ONNX test models at exactly the parameter limit boundary.

Creates two models:
  - param_limit_exact.onnx:  exactly 250,000 params (should PASS validation)
  - param_limit_over.onnx:   exactly 250,001 params (should FAIL -- 4 bytes over)

Strategy: export a base model (Linear(224,1096) + ReLU + Linear(1096,3)) which
has 249,897 ONNX params (249,891 learnable + 6 overhead).  Then inject a dummy
initializer tensor of exactly the right size to hit the target.
"""
import torch
import torch.nn as nn
import onnx
import numpy as np
from onnx import TensorProto, helper
from pathlib import Path

OBS_SIZE = 224
ACTION_SIZE = 3
MAX_PARAMS = 250_000
H = 1096


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(OBS_SIZE, H)
        self.fc2 = nn.Linear(H, ACTION_SIZE)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def count_onnx_params(model):
    """Count params the way our Rust validator does."""
    total = 0
    for t in model.graph.initializer:
        dims = list(t.dims)
        if len(dims) == 0:
            total += 1
        else:
            n = 1
            for d in dims:
                n *= max(d, 1)
            total += n
    for node in model.graph.node:
        for attr in node.attribute:
            if attr.t is not None and attr.t.ByteSize() > 0:
                dims = list(attr.t.dims)
                if len(dims) == 0:
                    total += 1
                else:
                    n = 1
                    for d in dims:
                        n *= max(d, 1)
                    total += n
            for t in attr.tensors:
                dims = list(t.dims)
                if len(dims) == 0:
                    total += 1
                else:
                    n = 1
                    for d in dims:
                        n *= max(d, 1)
                    total += n
    return total


def add_padding_initializer(model, pad_size, name="_pad"):
    """Add a dummy initializer of the given size to the ONNX model."""
    pad_data = np.zeros(pad_size, dtype=np.float32)
    pad_tensor = helper.make_tensor(name, TensorProto.FLOAT, [pad_size], pad_data)
    model.graph.initializer.append(pad_tensor)


def export_base():
    """Export the base model and return the ONNX model object."""
    m = SimpleModel()
    m.eval()
    path = "/tmp/_param_limit_base.onnx"
    torch.onnx.export(
        m, torch.randn(1, OBS_SIZE), path,
        input_names=["observation"], output_names=["action"],
        dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
        opset_version=17, do_constant_folding=True, dynamo=False,
    )
    return onnx.load(path)


if __name__ == "__main__":
    out_dir = Path(__file__).parent / "fixtures"
    out_dir.mkdir(exist_ok=True)

    base = export_base()
    base_params = count_onnx_params(base)
    print(f"Base model params: {base_params}")

    # --- Exact: 250,000 params (should PASS) ---
    exact = onnx.load("/tmp/_param_limit_base.onnx")
    pad_exact = MAX_PARAMS - base_params
    print(f"Padding exact model with {pad_exact} elements")
    add_padding_initializer(exact, pad_exact)
    exact_path = out_dir / "param_limit_exact.onnx"
    onnx.save(exact, str(exact_path))
    final_exact = count_onnx_params(onnx.load(str(exact_path)))
    print(f"  -> {final_exact} params (expect {MAX_PARAMS})")
    assert final_exact == MAX_PARAMS, f"Expected {MAX_PARAMS}, got {final_exact}"

    # --- Over: 250,001 params (should FAIL, 4 bytes over) ---
    over = onnx.load("/tmp/_param_limit_base.onnx")
    pad_over = MAX_PARAMS + 1 - base_params
    print(f"Padding over model with {pad_over} elements")
    add_padding_initializer(over, pad_over)
    over_path = out_dir / "param_limit_over.onnx"
    onnx.save(over, str(over_path))
    final_over = count_onnx_params(onnx.load(str(over_path)))
    print(f"  -> {final_over} params (expect {MAX_PARAMS + 1})")
    assert final_over == MAX_PARAMS + 1, f"Expected {MAX_PARAMS + 1}, got {final_over}"

    # Print summary
    sz_exact = exact_path.stat().st_size
    sz_over = over_path.stat().st_size
    print(f"\n  {exact_path.name}: {sz_exact} bytes, {final_exact} params -> should PASS")
    print(f"  {over_path.name}:  {sz_over} bytes, {final_over} params -> should FAIL")
    print(f"  File size difference: {sz_over - sz_exact} bytes")
    print("\nDone!")
