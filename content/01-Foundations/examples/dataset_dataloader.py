"""Build overlapping next-token windows and batch them with DataLoader."""

import torch
from torch.utils.data import DataLoader, Dataset


class NextTokenWindowDataset(Dataset):
    def __init__(self, token_ids, context_length):
        if context_length < 1:
            raise ValueError("context_length must be positive")
        if len(token_ids) <= context_length:
            raise ValueError("need at least context_length + 1 token IDs")
        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.context_length = context_length

    def __len__(self):
        return len(self.token_ids) - self.context_length

    def __getitem__(self, index):
        start = index
        end = start + self.context_length
        inputs = self.token_ids[start:end]
        targets = self.token_ids[start + 1 : end + 1]
        return inputs, targets


def main():
    dataset = NextTokenWindowDataset([0, 1, 2, 3, 4, 5], context_length=2)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    print("dataset windows:", len(dataset))
    for batch_number, (input_ids, target_ids) in enumerate(loader, start=1):
        print(f"batch {batch_number} inputs: {input_ids.tolist()}")
        print(f"batch {batch_number} targets: {target_ids.tolist()}")
        print(
            f"batch {batch_number} shapes: "
            f"inputs={tuple(input_ids.shape)}, targets={tuple(target_ids.shape)}"
        )


if __name__ == "__main__":
    main()
