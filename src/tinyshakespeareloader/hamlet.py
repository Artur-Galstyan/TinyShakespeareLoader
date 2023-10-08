import logging
import os
import sys
from typing import Callable, Optional
from jaxtyping import Array
import numpy as np
from torch.utils.data import DataLoader, Dataset

__author__ = "Artur A. Galstyan"
__copyright__ = "Artur A. Galstyan"
__license__ = "MIT"

_logger = logging.getLogger(__name__)


class MiniShakesPeare(Dataset):
    def __init__(self, data, block_size=8) -> None:
        super().__init__()
        self.block_size = block_size
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if index == -1:
            index = len(self.data) - 1
        x = self.data[index : index + self.block_size]
        y = self.data[index + 1 : index + self.block_size + 1]

        if index + self.block_size + 1 > len(self.data):
            diff = index + self.block_size + 1 - len(self.data)

            to_add_on_x = diff - 1
            to_add_on_y = diff

            x = np.concatenate((x, self.data[:to_add_on_x]))
            y = np.concatenate((y, self.data[:to_add_on_y]))

        return x, y


class TinyShakespeare:
    train_dataloader: DataLoader
    test_dataloader: DataLoader
    vocab_size: int | None
    encode: Callable[[str], Array] | None
    decode: Callable[[Array], str] | None

    def __init__(
        self,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        vocab_size: int | None,
        encode: Callable[[str], Array] | None,
        decode: Callable[[Array], str] | None,
    ):
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.vocab_size = vocab_size
        self.encode = encode
        self.decode = decode


def get_data(
    batch_size=4,
    train_ratio=0.9,
    block_size=8,
    encoder: Optional[Callable] = None,
    shuffle=False,
) -> TinyShakespeare:
    """Get the train and test dataloaders as well as the vocabulary size, the
    vocabulary itself, the encoding and decoding functions.
    The data is downloaded from the internet if it is not present in the current
    directory. Furthermore, the data is one hot encoded. You can also provide your
    own encoder function. In that case the vocabulary size is None as well as the
    encoding and decoding functions (since they are provided by you).

    Args:
        batch_size (int, optional): The batch size. Defaults to 4.
        train_ratio (float, optional): The ratio of the training data. Defaults to 0.9.
        block_size (int, optional): The size of the block. Defaults to 8.
        encoder (Optional[Callable], optional): The encoder function. Defaults to None, which performs a character level encoding.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to False.

    Returns:
        TinyShakespeare: The train and test dataloaders as well as the vocabulary size, the
        vocabulary itself, the encoding and decoding functions (if the encoder is None)
    """
    text = get_text()
    vocab_size = None
    decoder = None

    if encoder is not None:
        data = np.array(encoder(text))
    else:
        chars = sorted(list(set(text)))
        vocab_size = len(chars)

        # Lookup table to map single characters to integers
        char_to_idx = {ch: i for i, ch in enumerate(chars)}
        # Lookup table to map integers to single characters
        idx_to_char = {i: ch for i, ch in enumerate(chars)}

        def encode(string: str) -> Array:
            return np.array([char_to_idx[ch] for ch in string])  # type: ignore

        def decode(latent) -> str:
            return "".join([idx_to_char[idx] for idx in latent])

        encoder = encode
        decoder = decode
        data = np.array(encode(text))

    n = int(train_ratio * len(data))

    train_data = data[:n]
    test_data = data[n:]

    train_dataset = MiniShakesPeare(train_data, block_size=block_size)

    test_dataset = MiniShakesPeare(test_data, block_size=block_size)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    return TinyShakespeare(
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        vocab_size=vocab_size,
        encode=encoder,
        decode=decoder,
    )


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=loglevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )


def get_text():
    # get current absolute path to this file
    current_path = os.path.abspath(os.path.dirname(__file__))
    # get the parent directory of the current path
    parent_path = os.path.dirname(current_path)

    # check if there is a folder called data in the parent_path

    if not os.path.exists(parent_path + "/data"):
        os.makedirs(parent_path + "/data")

    if not os.path.exists(parent_path + "/data/input.txt"):
        import urllib.request

        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"  # noqa
        _logger.info("Downloading the dataset from %s", url)
        urllib.request.urlretrieve(url, parent_path + "/data/input.txt")

    with open(parent_path + "/data/input.txt", "r") as f:
        text = f.read()
    return text
