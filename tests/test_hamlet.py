from torch.utils.data import DataLoader

from tinyshakespeareloader.hamlet import get_data

__author__ = "Artur A. Galstyan"
__copyright__ = "Artur A. Galstyan"
__license__ = "MIT"


def test_dataloaders():
    """API Tests"""
    data = get_data()
    train_data, val_data = data["train_dataloader"], data["test_dataloader"]

    # check if the dataloaders are not None
    assert train_data is not None and val_data is not None

    # check if the dataloaders are of type DataLoader
    assert isinstance(train_data, DataLoader) and isinstance(val_data, DataLoader)


def test_dataloaders_batch_size():
    """API Tests"""
    data = get_data()
    train_data, val_data = data["train_dataloader"], data["test_dataloader"]

    # check if the dataloaders have the correct batch size
    assert train_data.batch_size == 4 and val_data.batch_size == 4


def test_dataloaders_iterability():
    """API Tests"""
    train_ratio = 0.9
    batch_size = 4
    block_size = 8
    data = get_data(
        train_ratio=train_ratio, batch_size=batch_size, block_size=block_size
    )
    train_data, val_data = data["train_dataloader"], data["test_dataloader"]
    vocab_size = data["vocabulary_size"]

    assert vocab_size == 65

    # check if the dataloaders are iterable
    assert iter(train_data) is not None and iter(val_data) is not None

    for x, y in train_data:
        assert x.shape[1] == block_size and y.shape[1] == block_size


'''
def test_main(capsys):
    """CLI Tests"""
    # capsys is a pytest fixture that allows asserts against stdout/stderr
    # https://docs.pytest.org/en/stable/capture.html
    main(["7"])
    captured = capsys.readouterr()
    assert "The 7-th Fibonacci number is 13" in captured.out
'''
