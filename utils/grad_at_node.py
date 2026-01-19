#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Given:
  - a PyTorch model class (from --model_py / --model_class)
  - one .npz file that contains:
      * input array (key via --input_key, or auto-detect)
      * weights arrays (keys matching model.state_dict() keys)
  - a module node name (e.g. "sep_conv2")
Compute:
  - grad_x: d(target)/d(input)
  - grad_w_*: d(target)/d(param) for params directly in that module (recurse=False)
Save:
  - out.npz with grad_x, grad_w_*, meta

python /app/Onnx4Deeploy/utils/grad_at_node.py \
  --model_py /app/Onnx4Deeploy/Tests/Models/MI-BMInet/mi_bminet_model/mi_bminet.py \
  --model_class MIBMINetDeploy \
  --model_kwargs_json '{"C":8,"T":2000,"N":4}' \
  --npz /app/Onnx4Deeploy/Tests/Models/MI-BMInet/onnx/MI_BMINet_train_C8_T2000_F1_8_D2_N4_fc_sep_conv2/inputs.npz \
  --node sep_conv2 \
  --loss crossentropy \
  --out_npz out_grad.npz


Target:
  - default: target = node_output.sum()
  - if --index provided: target = node_output[index] (scalar; if not scalar -> sum)
Optimizer:
  - optional optimizer; optional --step to apply one update
"""

import argparse
import importlib.util
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


# ----------------- import helpers -----------------

def import_from_path(py_path: str):
    py_path = os.path.abspath(py_path)
    mod_name = os.path.splitext(os.path.basename(py_path))[0]
    spec = importlib.util.spec_from_file_location(mod_name, py_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from: {py_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return module


# ----------------- npz helpers -----------------

def load_npz(path: str) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def pick_input_key(npz_dict: Dict[str, np.ndarray], input_key: Optional[str]) -> str:
    if input_key is not None:
        if input_key not in npz_dict:
            raise KeyError(f"--input_key '{input_key}' not found. Keys: {list(npz_dict.keys())}")
        return input_key

    # auto-detect common names
    for k in ["x", "input", "inputs", "data"]:
        if k in npz_dict:
            return k

    # fallback: choose the first array that does NOT look like a weight key
    # (rough heuristic: weights often contain '.' and end with weight/bias/running_mean/...)
    for k in npz_dict.keys():
        if "." not in k:
            return k

    # last resort: first key
    return next(iter(npz_dict.keys()))


def np_array_to_tensor(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    # Keep dtype as stored; PyTorch will accept float/half/int
    t = torch.from_numpy(arr)
    return t.to(device=device)


def load_state_dict_from_npz(model: torch.nn.Module, npz_dict: Dict[str, np.ndarray], strict: bool = False):
    """
    Load weights from npz where keys match model.state_dict() keys.
    Ignores keys not in state_dict.
    """
    sd = model.state_dict()
    loadable = {}
    for k in sd.keys():
        if k in npz_dict:
            loadable[k] = torch.from_numpy(npz_dict[k])

    missing = [k for k in sd.keys() if k not in loadable]
    used = list(loadable.keys())
    unexpected = [k for k in npz_dict.keys() if k in sd.keys() is False]

    # load into a copy of state_dict (handles buffers too)
    new_sd = {}
    for k, v in sd.items():
        if k in loadable:
            w = loadable[k]
            if w.shape != v.shape:
                raise RuntimeError(f"Shape mismatch for '{k}': npz {tuple(w.shape)} vs model {tuple(v.shape)}")
            # preserve dtype/device later via load_state_dict + to(device)
            new_sd[k] = w
        else:
            new_sd[k] = v

    # load
    model.load_state_dict(new_sd, strict=False)

    if strict and missing:
        raise RuntimeError(f"Missing keys in npz (strict=True): {missing}")

    return {
        "used_keys": used,
        "missing_keys": missing,
        "unexpected_npz_keys_count": len(unexpected),
    }


# ----------------- other helpers -----------------

def parse_index(index_str: Optional[str]) -> Optional[Tuple[int, ...]]:
    if index_str is None:
        return None
    parts = [p.strip() for p in index_str.split(",") if p.strip() != ""]
    if not parts:
        return None
    return tuple(int(p) for p in parts)


def build_optimizer(
    opt_name: str,
    params: List[torch.nn.Parameter],
    lr: float,
    momentum: float,
    weight_decay: float,
):
    opt_name = opt_name.lower()
    if opt_name == "none":
        return None
    if opt_name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if opt_name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if opt_name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {opt_name}")


# ----------------- main -----------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model_py", required=True, help="Path to python file defining the model class")
    ap.add_argument("--model_class", required=True, help="Class name, e.g. MIBMINetDeploy")
    ap.add_argument(
        "--model_kwargs_json",
        default=None,
        help="JSON for constructor kwargs, e.g. '{\"C\":8,\"T\":2000,\"N\":4}'",
    )

    ap.add_argument("--npz", required=True, help="Path to .npz containing BOTH input and weights")
    ap.add_argument("--input_key", default=None, help="Key name for input array inside npz (optional)")
    ap.add_argument(
        "--weights_strict",
        action="store_true",
        help="If set, require that ALL model state_dict keys exist in npz (usually too strict).",
    )

    ap.add_argument("--node", required=True, help="Module path to hook, e.g. 'sep_conv2'")
    ap.add_argument("--index", default=None, help="Optional element index, e.g. '0,0,0,10'")
    ap.add_argument("--loss", default="node_sum", choices=["node_sum", "crossentropy"],
                    help="Loss function: node_sum (default) or crossentropy")

    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--dtype", default=None, choices=["float16", "float32", "float64"],
                    help="Force input dtype. If omitted, keep npz dtype.")

    ap.add_argument("--optimizer", default="none", choices=["none", "sgd", "adam", "adamw"])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--step", action="store_true", help="If set, run optimizer.step() after backward")

    ap.add_argument("--out_npz", required=True, help="Output npz file path")

    args = ap.parse_args()

    # import model
    module = import_from_path(args.model_py)
    if not hasattr(module, args.model_class):
        raise AttributeError(f"Class '{args.model_class}' not found in {args.model_py}")
    ModelCls = getattr(module, args.model_class)

    model_kwargs = {}
    if args.model_kwargs_json:
        model_kwargs = json.loads(args.model_kwargs_json)
        if not isinstance(model_kwargs, dict):
            raise ValueError("--model_kwargs_json must decode to a JSON object/dict.")

    model = ModelCls(**model_kwargs)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    model = model.to(device)

    # load npz
    npz_dict = load_npz(args.npz)

    # load weights from npz (keys must match state_dict keys)
    weight_report = load_state_dict_from_npz(model, npz_dict, strict=args.weights_strict)

    # choose input
    input_key = pick_input_key(npz_dict, args.input_key)
    x_np = npz_dict[input_key]

    x = np_array_to_tensor(x_np, device=device)
    if args.dtype is not None:
        dtype_map = {"float16": torch.float16, "float32": torch.float32, "float64": torch.float64}
        x = x.to(dtype=dtype_map[args.dtype])
    x.requires_grad_(True)

    # load labels if available (for crossentropy loss)
    labels = None
    if "labels" in npz_dict:
        labels_np = npz_dict["labels"]
        labels = torch.from_numpy(labels_np).to(device)
    elif args.loss == "crossentropy":
        raise RuntimeError("--loss=crossentropy requires 'labels' key in npz file")

    # locate module
    try:
        target_module = model.get_submodule(args.node)
    except Exception as e:
        all_names = [name for name, _ in model.named_modules()]
        raise RuntimeError(
            f"Cannot find node '{args.node}'. Node must be a module path from model.named_modules().\n"
            f"Example names: {all_names[:30]}{' ...' if len(all_names) > 30 else ''}"
        ) from e

    # hook output and input
    captured: Dict[str, torch.Tensor] = {}

    def hook_fn(_m, inp, out):
        # Capture output
        node_output = out[0] if isinstance(out, (tuple, list)) else out
        # Retain gradient for node output
        if node_output.requires_grad:
            node_output.retain_grad()
        captured["node_output"] = node_output
        # Capture input (first input tensor)
        node_input = inp[0] if isinstance(inp, (tuple, list)) else inp
        # Retain gradient for node input
        if node_input.requires_grad:
            node_input.retain_grad()
        captured["node_input"] = node_input

    h = target_module.register_forward_hook(hook_fn)

    # optimizer (optional)
    opt = build_optimizer(
        args.optimizer,
        list(model.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    model.train()  # optimizer 相关时更符合训练语义（你这个 deploy 模型也无 BN/dropout，无所谓）

    # clear grads
    model.zero_grad(set_to_none=True)
    if opt is not None:
        opt.zero_grad(set_to_none=True)
    if x.grad is not None:
        x.grad = None

    # forward
    final_output = model(x)
    h.remove()

    if "node_output" not in captured:
        raise RuntimeError("Hook did not capture node output. Does the node execute in forward()?")

    node_out = captured["node_output"]

    # build target
    if args.loss == "crossentropy":
        if labels is None:
            raise RuntimeError("CrossEntropy loss requires labels, but labels not found in npz")
        criterion = torch.nn.CrossEntropyLoss()
        target = criterion(final_output, labels)
        target_desc = f"CrossEntropyLoss(final_output, labels)"
    else:
        # Default: node_sum with optional index
        idx = parse_index(args.index)
        if idx is None:
            target = node_out.sum()
            target_desc = "sum(node_output)"
        else:
            try:
                target = node_out[idx]
            except Exception as e:
                raise RuntimeError(f"Index {idx} failed for node_output.shape={tuple(node_out.shape)}") from e
            if target.numel() != 1:
                target = target.sum()
            target_desc = f"node_output[{idx}]"

    # backward to populate grads
    target.backward()

    if x.grad is None:
        raise RuntimeError("x.grad is None. Ensure x.requires_grad=True and target depends on x.")
    grad_x = x.grad.detach().cpu().numpy()

    # Extract gradient of node input (∂Loss/∂(node_input))
    grad_node_input = None
    if "node_input" in captured and captured["node_input"].grad is not None:
        grad_node_input = captured["node_input"].grad.detach().cpu().numpy()

    # Extract gradient of node output (∂Loss/∂(node_output))
    grad_node_output = None
    if "node_output" in captured and captured["node_output"].grad is not None:
        grad_node_output = captured["node_output"].grad.detach().cpu().numpy()

    # Extract actual data values (forward pass)
    node_input_data = None
    if "node_input" in captured:
        node_input_data = captured["node_input"].detach().cpu().numpy()

    node_output_data = None
    if "node_output" in captured:
        node_output_data = captured["node_output"].detach().cpu().numpy()

    # grad_w only for params directly in that module
    out: Dict[str, Any] = {"grad_x": grad_x}
    if grad_node_input is not None:
        out["grad_node_input"] = grad_node_input
    if grad_node_output is not None:
        out["grad_node_output"] = grad_node_output
    if node_input_data is not None:
        out["node_input_data"] = node_input_data
    if node_output_data is not None:
        out["node_output_data"] = node_output_data
    grad_w_names: List[str] = []
    for n, p in target_module.named_parameters(recurse=False):
        if p.requires_grad:
            if p.grad is None:
                raise RuntimeError(f"Parameter '{args.node}.{n}' grad is None (disconnected?)")
            grad_w_names.append(n)
            out[f"grad_w_{n}"] = p.grad.detach().cpu().numpy()

    did_step = False
    if opt is not None and args.step:
        opt.step()
        did_step = True

    meta = {
        "model_py": os.path.abspath(args.model_py),
        "model_class": args.model_class,
        "model_kwargs": model_kwargs,
        "npz": os.path.abspath(args.npz),
        "input_key": input_key,
        "weights_loading": weight_report,
        "node": args.node,
        "loss": args.loss,
        "target": target_desc,
        "node_input_shape": tuple(captured["node_input"].shape) if "node_input" in captured else None,
        "node_output_shape": tuple(node_out.shape),
        "final_output_shape": tuple(final_output.shape),
        "labels_shape": tuple(labels.shape) if labels is not None else None,
        "grad_x_shape": tuple(grad_x.shape),
        "grad_node_input_shape": tuple(grad_node_input.shape) if grad_node_input is not None else None,
        "grad_node_output_shape": tuple(grad_node_output.shape) if grad_node_output is not None else None,
        "node_input_data_shape": tuple(node_input_data.shape) if node_input_data is not None else None,
        "node_output_data_shape": tuple(node_output_data.shape) if node_output_data is not None else None,
        "grad_w_names": grad_w_names,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "momentum": args.momentum if args.optimizer == "sgd" else None,
        "weight_decay": args.weight_decay,
        "did_step": did_step,
        "device": str(device),
        "dtype_forced": args.dtype,
    }
    out["meta"] = np.array(json.dumps(meta), dtype=object)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_npz)) or ".", exist_ok=True)
    np.savez(args.out_npz, **out)

    print("Saved:", args.out_npz)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
