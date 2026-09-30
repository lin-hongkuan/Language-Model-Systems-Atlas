"""A small, CPU-only demonstration of tensors, autograd, and nn.Module."""

import torch
from torch import nn
from torch.nn import functional as F


class TinyClassifier(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=4, num_classes=3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features):
        return self.layers(features)


def main():
    # Tensor: a typed, shaped array. This one has two rows and three columns.
    ids = torch.tensor([[0, 1, 2], [2, 1, 0]], dtype=torch.long)
    print("ids shape:", tuple(ids.shape))
    print("ids dtype:", ids.dtype)

    # Autograd records operations that use a tensor requiring gradients.
    x = torch.tensor(2.0, requires_grad=True)
    y = x**2 + 3 * x
    y.backward()
    print("y at x=2:", y.item())
    print("dy/dx at x=2:", x.grad.item())

    # nn.Module owns layers/parameters and defines a forward computation.
    torch.manual_seed(0)
    model = TinyClassifier()
    features = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    )
    labels = torch.tensor([0, 1, 2, 1], dtype=torch.long)
    logits = model(features)
    print("features shape:", tuple(features.shape))
    print("logits shape:", tuple(logits.shape))

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    before = next(model.parameters()).detach().clone()
    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    optimizer.step()
    changed = not torch.equal(before, next(model.parameters()).detach())
    print("loss is finite:", bool(torch.isfinite(loss)))
    print("parameter changed after step:", changed)


if __name__ == "__main__":
    main()
