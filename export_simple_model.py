"""Export a simple PyTorch linear+relu model to Torch Dialect MLIR."""
import torch
from torch_mlir import fx

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 3)

    def forward(self, x):
        return torch.relu(self.linear(x))

model = SimpleModel()
model.eval()
example = torch.randn(1, 2)

result = fx.export_and_import(model, example, output_type="torch")
mlir_text = result.operation.get_asm()
print(mlir_text)

with open("simple_model.mlir", "w") as f:
    f.write(mlir_text)
print("\n[OK] Written to simple_model.mlir")
