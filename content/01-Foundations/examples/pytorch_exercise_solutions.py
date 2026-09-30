"""Runnable worked answers for shape, gradient, and loss exercises."""

import torch
from torch import nn
from torch.nn import functional as F


def main():
    batch_size, seq_len, vocab_size, embedding_dim = 2, 3, 5, 4
    input_ids = torch.tensor([[0, 1, 2], [2, 3, 4]], dtype=torch.long)
    targets = torch.tensor([[1, 2, 3], [3, 4, 0]], dtype=torch.long)
    embedding = nn.Embedding(vocab_size, embedding_dim)
    projection = nn.Linear(embedding_dim, vocab_size)

    vectors = embedding(input_ids)
    logits = projection(vectors)
    loss = F.cross_entropy(
        logits.reshape(batch_size * seq_len, vocab_size),
        targets.reshape(batch_size * seq_len),
    )
    print("input shape:", tuple(input_ids.shape))
    print("embedding shape:", tuple(vectors.shape))
    print("logits shape:", tuple(logits.shape))
    print("flattened logits shape:", (batch_size * seq_len, vocab_size))
    print("flattened targets shape:", (batch_size * seq_len,))
    print("loss is finite:", bool(torch.isfinite(loss)))

    x = torch.tensor(2.0, requires_grad=True)
    (x**2 + 3 * x).backward()
    print("gradient of x**2 + 3*x at x=2:", x.grad.item())


if __name__ == "__main__":
    main()
