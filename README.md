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


    tinyshakespeare = get_data()

    train_dataloader, test_dataloader = tinyshakespeare.train_dataloader, tinyshakespeare.test_dataloader

    for batch in train_dataloader:
        print(batch)

```

You can provide your own encoder function if you want - but it's not required! If you don't, it will just simply use the character level encoding, that Andrej also used in his tutorial.
