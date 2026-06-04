"""W3: Batch-export simple PyTorch models to Torch Dialect MLIR.

Usage:
    conda activate novel_llm
    python scripts/export_simple_model.py
"""
import sys
from pathlib import Path

import torch
from torch_mlir import fx

# Add project root to path for model imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.simple_models import MODELS  # noqa: E402

OUTPUT_DIR = Path("mlir/exported")


def export_model(name: str, model_cls: type) -> str:
    """Export a PyTorch model to Torch Dialect MLIR, return the MLIR text."""
    model = model_cls()
    model.eval()
    example = model.example_input

    result = fx.export_and_import(model, example, output_type="torch")
    return result.operation.get_asm()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, model_cls in MODELS.items():
        print(f"[{name}] exporting...", end=" ", flush=True)
        try:
            mlir_text = export_model(name, model_cls)
        except Exception as e:
            print(f"FAILED: {e}")
            continue

        out_path = OUTPUT_DIR / f"{name}.mlir"
        out_path.write_text(mlir_text)
        n_lines = mlir_text.count("\n") + 1
        n_funcs = mlir_text.count("func.func @")
        print(f"OK → {out_path} ({n_lines} lines, {n_funcs} functions)")

    print(f"\nAll exports written to {OUTPUT_DIR.resolve()}/")


if __name__ == "__main__":
    main()
