"""A runnable tour of the Python syntax used in the course notes."""


def encode(tokens, vocabulary):
    """Convert a list of tokens to their integer IDs."""
    token_ids = []
    for token in tokens:
        if token not in vocabulary:
            raise ValueError(f"unknown token: {token}")
        token_ids.append(vocabulary[token])
    return token_ids


def make_next_token_example(token_ids):
    """Return a prefix and its one-token-shifted targets."""
    inputs = token_ids[:-1]
    targets = token_ids[1:]
    return inputs, targets


def main():
    sentence = "猫 喜欢 睡觉"
    tokens = sentence.split()
    vocabulary = {"猫": 0, "喜欢": 1, "睡觉": 2}
    token_ids = encode(tokens, vocabulary)

    print("sentence:", sentence)
    print("tokens:", tokens)
    print("token_ids:", token_ids)
    print("first token:", tokens[0])
    print("last two tokens:", tokens[1:])

    # A dict stores named fields; a list stores an ordered sequence.
    batch = {"input_ids": token_ids[:-1], "labels": token_ids[1:]}
    print("batch:", batch)

    # if filters data; for processes each item; append grows the output list.
    even_ids = []
    for token_id in token_ids:
        if token_id % 2 == 0:
            even_ids.append(token_id)
    print("even token ids:", even_ids)

    inputs, targets = make_next_token_example([0, 1, 2, 3])
    print("next-token inputs:", inputs)
    print("next-token targets:", targets)

    try:
        encode(["未知词"], vocabulary)
    except ValueError as error:
        print("caught expected error:", error)


if __name__ == "__main__":
    main()
