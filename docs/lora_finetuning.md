# LoRA Fine-Tuning

Both `TinyTransformer` and `CCT` support parameter-efficient fine-tuning via
[LoRA](https://arxiv.org/abs/2106.09685). Low-rank `A`/`B` adapters are attached
to the four projections of the attention block (`q_proj`, `k_proj`, `v_proj`,
`out_proj`); the existing training-graph generator (ORT `generate_artifacts`)
then produces a backward graph containing gradients **only** for those adapters.

## Forward

For each attention projection:

```
y = W·x  +  (x · A · B) · (alpha / r)
```

with `A ∈ ℝ^(D×r)`, `B ∈ ℝ^(r×D)`. `A` is kaiming-uniform-initialised, `B` is
zero-initialised, so at step 0 the LoRA delta is zero and the model output
matches the non-LoRA baseline.

The path is implemented as `Reshape → 2-D MatMul → 2-D MatMul → Reshape` so the
exported graph contains no `perm=[1,0]` Transpose nodes — same pattern as
`_lin3d`.

## Configuration

| key              | default | description                                  |
| ---------------- | ------- | -------------------------------------------- |
| `use_lora`       | `False` | Attach LoRA adapters to the attention block  |
| `lora_r`         | `4`     | LoRA rank                                    |
| `lora_alpha`     | `16`    | Effective scale = `alpha / r`                |

`training_strategy="lora"` freezes every parameter whose name does not contain
`lora_`, leaving only the eight `lora_{q,k,v,out}_{A,B}` matrices trainable.

## Usage

```python
from onnx4deeploy.models import TinyTransformerExporter

exp = TinyTransformerExporter(save_path="./out")
exp._config_overrides = {
    "use_lora": True,
    "training_strategy": "lora",
    "lora_r": 4,
    "lora_alpha": 16,
}
exp.export_training()
```

The resulting `network_train.onnx` exposes 8 trainable tensors and 8
`InPlaceAccumulatorV2` nodes (one pair per LoRA matrix); all base weights remain
as frozen graph inputs.

## CCT (CLI usage)

```bash
python Onnx4Deeploy.py -model CCT -mode train --use-lora \
    --n-steps 16 --n-accum 2
```

Then run the generated graph through Deeploy:

```bash
python deeployTrainingRunner_tiled_siracusa.py \
    -t /path/to/onnx/model/cct_lora_train \
    --l1=128000 --defaultMemLevel=L3 \
    --n-steps=16 --n-accum=2
```

If `inputs.npz` only contains a single mini-batch (i.e. `--n-batches 1
--n-accum 1`), pass `--num-data-inputs=1` explicitly — the runner cannot
auto-detect it without `mb1_arr_*` entries.

## LoRA-specific Deeploy front-end fixes

LoRA freezes most weights; the resulting graph triggers two front-end issues
that the optimizer pipeline now handles automatically:

1. **N-input Sum → chained Add value_info.** With LoRA adapters, the gradient
   at a residual feed-point has more than 3 contributors (e.g. `q_proj`,
   `k_proj`, `v_proj` plus their LoRA branches). ORT emits an N-input Sum;
   `convert_sum_to_add` (in `trainOptimization.py`) now stamps `value_info` on
   every `_intermediate_{j}` it creates so Deeploy's shape assertion passes.

2. **Constant-fed multi-consumer Transpose/Reshape.** Frozen weights become
   `Constant` nodes; `Constant → Transpose → {fwd MatMul, bwd Gemm}` would be
   folded by Deeploy into a single Constant with two consumers, tripping
   `DeeployTypes.hoistConstant`'s `len(constant.outputs) <= 1` assertion. The
   `duplicate_constant_fed_transposes` pass (in `graph_cleaner.py`, wired into
   the train pipeline after `process_onnx_model_name_with_type`) duplicates any
   single-output node whose inputs are all from `Constant` nodes/initializers
   and whose output has more than one consumer.
