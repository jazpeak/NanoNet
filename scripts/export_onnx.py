import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
model.load_state_dict(torch.load("mnist_mlp.pt", map_location="cpu"))
model.eval()

dummy = torch.randn(1, 1, 28, 28)

torch.onnx.export(
    model,
    dummy,
    "mnist_mlp.onnx",
    input_names=["X"],
    output_names=["Y"],
    opset_version=13
)

print("Exported mnist_mlp.onnx")
