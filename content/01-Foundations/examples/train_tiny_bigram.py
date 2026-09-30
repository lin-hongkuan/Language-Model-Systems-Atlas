"""Train a tiny next-token model on a repeating three-character corpus.

This CPU-only example deliberately uses a one-token context. It teaches the
data/model/loss/update loop; it is not a Transformer and downloads no data.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


class NextTokenDataset(Dataset):
    def __init__(self, token_ids):
        self.inputs = torch.tensor(token_ids[:-1], dtype=torch.long)
        self.targets = torch.tensor(token_ids[1:], dtype=torch.long)

    def __len__(self):
        return self.inputs.numel()

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]


class TinyBigramLM(nn.Module):
    """Use the current token embedding to predict the next token."""

    def __init__(self, vocab_size, embedding_dim=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, input_ids):
        # input_ids [B] -> vectors [B, D] -> next-token logits [B, V]
        vectors = self.embedding(input_ids)
        return self.output(vectors)


def main():
    torch.manual_seed(7)
    torch.set_num_threads(1)

    corpus = "猫追球" * 32
    vocabulary = {character: index for index, character in enumerate(sorted(set(corpus)))}
    token_ids = [vocabulary[character] for character in corpus]
    dataset = NextTokenDataset(token_ids)
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        generator=torch.Generator().manual_seed(7),
    )

    model = TinyBigramLM(vocab_size=len(vocabulary))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    all_inputs = dataset.inputs
    all_targets = dataset.targets

    with torch.no_grad():
        initial_loss = F.cross_entropy(model(all_inputs), all_targets).item()

    for _epoch in range(80):
        model.train()
        for input_ids, target_ids in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids)
            loss = F.cross_entropy(logits, target_ids)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        final_loss = F.cross_entropy(model(all_inputs), all_targets).item()
        predictions = {}
        for character, token_id in vocabulary.items():
            context = torch.tensor([token_id], dtype=torch.long)
            next_id = model(context).argmax(dim=-1).item()
            next_character = next(char for char, idx in vocabulary.items() if idx == next_id)
            predictions[character] = next_character

    print("vocabulary:", vocabulary)
    print("training examples:", len(dataset))
    print(f"loss decreased: {initial_loss:.3f} -> {final_loss:.3f}")
    for character in sorted(predictions):
        print(f"{character} -> {predictions[character]}")


if __name__ == "__main__":
    main()
