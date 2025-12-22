import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Data
transform = transforms.ToTensor()
train_ds = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_ds  = datasets.MNIST("./data", train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_ds, batch_size=256)

# Model (ONLY ops your engine supports)
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28*28, 128),
    nn.ReLU(),
    nn.Linear(128, 10)   # logits (no softmax)
)

opt = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Train
model.train()
for epoch in range(3):
    for x, y in train_loader:
        opt.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
    print(f"epoch {epoch} done")

# Eval
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for x, y in test_loader:
        logits = model(x)
        pred = torch.argmax(logits, dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
print("Test accuracy:", correct/total)

torch.save(model.state_dict(), "mnist_mlp.pt")
print("Saved mnist_mlp.pt")
