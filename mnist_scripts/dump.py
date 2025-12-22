import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms

# Load model
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
model.load_state_dict(torch.load("mnist_mlp.pt", map_location="cpu"))
model.eval()

ds = datasets.MNIST("./data", train=False, download=True, transform=transforms.ToTensor())

N = 100  # dump first 100 test images
X = []
Y = []
LOGITS = []

with torch.no_grad():
    for i in range(N):
        img, label = ds[i]
        logits = model(img.unsqueeze(0)).numpy().reshape(-1)
        X.append(img.numpy().reshape(-1))   # 784
        Y.append(label)
        LOGITS.append(logits)

np.savetxt("mnist_X.txt", np.array(X))
np.savetxt("mnist_Y.txt", np.array(Y), fmt="%d")
np.savetxt("mnist_logits_ref.txt", np.array(LOGITS))

print("Saved mnist_X.txt, mnist_Y.txt, mnist_logits_ref.txt")
