
# TinyShakespeareLoader

## A PyTorch DataLoader for the TinyShakespeare Dataset

If you followed Andrej Karparthy's tutorial on GPT, you will notice he used the TinyShakespeare dataset, but not with the PyTorch DataLoader.
This repository fills that gap.

The TinyShakespeare dataset is a small dataset of Shakespeare's plays, with each line as a separate sample. To install this package, simply run:

```console

    pip install TinyShakespeareLoader

```

Then, to use it, simply import it and use it as a PyTorch DataLoader:

```python
    from TinyShakespeareLoader.hamlet import get_data


    data = get_data()

    train_dataloader, test_dataloader = data["train_dataloader"], data["test_dataloader"]

    for batch in train_dataloader:
        print(batch)

```
