"""Runnable worked answers for the Python foundation exercises."""


def encode(tokens, vocabulary, unknown_id=None):
    """Encode tokens; use the reserved unknown_id only when supplied."""
    token_ids = []
    for token in tokens:
        if token in vocabulary:
            token_ids.append(vocabulary[token])
        elif unknown_id is not None:
            token_ids.append(unknown_id)
        else:
            raise ValueError(f"unknown token: {token}")
    return token_ids


def count_tokens(tokens):
    counts = {}
    for token in tokens:
        if token not in counts:
            counts[token] = 0
        counts[token] += 1
    return counts


def make_next_token_example(token_ids):
    if len(token_ids) < 2:
        raise ValueError("need at least two IDs for a next-token example")
    return token_ids[:-1], token_ids[1:]


def main():
    vocabulary = {"猫": 0, "追": 1, "球": 2, "<unk>": 3}
    sentence_tokens = ["猫", "追", "猫", "球"]
    ids = encode(sentence_tokens, vocabulary)
    print("encoded:", ids)
    print("counts:", count_tokens(sentence_tokens))
    inputs, targets = make_next_token_example(ids)
    print("inputs:", inputs)
    print("targets:", targets)
    print("unknown fallback:", encode(["鸟"], vocabulary, unknown_id=3))


if __name__ == "__main__":
    main()
