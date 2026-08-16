"""Convert a Hugging Face PEFT LoRA adapter (from Google Colab) into the
MLX adapter format that server.py / chatbot.py / FintechFineTuner expect.

Usage:
    .venv/bin/python3.11 convert_peft_to_mlx.py <peft_dir> <mlx_dir>

The Colab trainer saves to a folder containing:
    adapter_config.json   (HF: r, lora_alpha, target_modules, ...)
    adapter_model.safetensors   (HF: ...lora_A.weight / lora_B.weight ...)

Output (MLX):
    adapter_config.json   (fine_tune_type / num_layers / lora_parameters)
    adapters.safetensors  (model.layers.{i}.{path}.lora_a / lora_b)
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
from safetensors.torch import load_file, save_file


def convert(peft_dir: str, mlx_dir: str):
    peft_dir = Path(peft_dir)
    mlx_dir = Path(mlx_dir)
    mlx_dir.mkdir(parents=True, exist_ok=True)

    with open(peft_dir / "adapter_config.json") as f:
        hf_cfg = json.load(f)

    r = int(hf_cfg.get("r", 16))
    lora_alpha = float(hf_cfg.get("lora_alpha", 32))
    dropout = float(hf_cfg.get("lora_dropout", 0.05))
    target_modules = hf_cfg.get("target_modules", [])
    num_layers = int(hf_cfg.get("num_layers", 28))

    # Map bare HF module names to full MLX paths.
    module_paths = {
        "q_proj": "self_attn.q_proj",
        "k_proj": "self_attn.k_proj",
        "v_proj": "self_attn.v_proj",
        "o_proj": "self_attn.o_proj",
        "gate_proj": "mlp.gate_proj",
        "up_proj": "mlp.up_proj",
        "down_proj": "mlp.down_proj",
    }
    full_keys = [
        module_paths[m] for m in target_modules if m in module_paths
    ]

    hf_weights = load_file(str(peft_dir / "adapter_model.safetensors"))

    mlx = {}
    for key, tensor in hf_weights.items():
        # HF key example:
        #   base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
        # Split off the lora part.
        head, dot, suffix = key.rpartition(".")
        head, _, lora_name = head.rpartition(".")
        if lora_name not in ("lora_A", "lora_B"):
            continue
        if suffix != "weight":
            continue

        m = head.split(".", 1)
        if m[0] == "base_model":
            rest = m[1]
            m2 = rest.split(".", 1)
            if m2[0] == "model":
                mlx_head = m2[1]
            else:
                mlx_head = rest
        else:
            mlx_head = head
        out = f"{mlx_head}.lora_{lora_name[-1].lower()}"

        # HF: A is (r, in), B is (out, r); MLX: a is (in, r), b is (r, out)
        if lora_name == "lora_A":
            mlx[out] = tensor.t().contiguous()
            in_dims = tensor.shape[1]
        else:
            mlx[out] = tensor.t().contiguous()
            out_dims = tensor.shape[0]

    save_file(mlx, str(mlx_dir / "adapters.safetensors"))

    mlx_cfg = {
        "fine_tune_type": "lora",
        "num_layers": num_layers,
        "lora_parameters": {
            "rank": r,
            "scale": lora_alpha / r,
            "dropout": dropout,
            "keys": full_keys,
        },
    }
    with open(mlx_dir / "adapter_config.json", "w") as f:
        json.dump(mlx_cfg, f, indent=2)

    # MLX loads adapters.safetensors with strict=False against its own file
    # naming, so we copy the tokenizer alongside (server.py needs it).
    for name in ("tokenizer_config.json", "tokenizer.json"):
        src = peft_dir / name
        if src.exists():
            shutil.copy(src, mlx_dir / name)

    print(f"Converted {len(mlx)} tensors:")
    print(f"  rank={r}, scale={lora_alpha / r}, dropout={dropout}")
    print(f"  layers={num_layers}, modules={target_modules}")
    print(f"\nWrote:\n  {mlx_dir / 'adapters.safetensors'}\n  {mlx_dir / 'adapter_config.json'}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])