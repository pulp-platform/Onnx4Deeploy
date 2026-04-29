# LoRA Fine-Tuning (TinyTransformer)

`TinyTransformer` supports parameter-efficient fine-tuning via
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
