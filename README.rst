.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

..    .. image:: https://readthedocs.org/projects/tinyShakespeareLoader/badge/?version=latest
        :alt: ReadTheDocs
        :target: https://tinyShakespeareLoader.readthedocs.io/en/stable/
..    .. image:: https://img.shields.io/coveralls/github/<USER>/tinyShakespeareLoader/main.svg
        :alt: Coveralls
        :target: https://coveralls.io/r/<USER>/tinyShakespeareLoader
.. image:: https://img.shields.io/pypi/v/tinyShakespeareLoader.svg
    :alt: PyPI-Server
    :target: https://pypi.org/project/tinyShakespeareLoader/
..    .. image:: https://img.shields.io/conda/vn/conda-forge/tinyShakespeareLoader.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/tinyShakespeareLoader
..    .. image:: https://pepy.tech/badge/tinyShakespeareLoader/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/tinyShakespeareLoader
..    .. image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter
        :alt: Twitter
        :target: https://twitter.com/tinyShakespeareLoader

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

=====================
TinyShakespeareLoader
=====================


    A PyTorch DataLoader for the TinyShakespeare Dataset


If you followed Andrej Karparthy's tutorial on GPT, you will notice he used the TinyShakespeare dataset, but not with the PyTorch DataLoader.
This repository fills that gap.

The TinyShakespeare dataset is a small dataset of Shakespeare's plays, with each line as a separate sample. To install this package, simply run:

.. code-block:: console
    $ pip install TinyShakespeareLoader

Then, to use it, simply import it and use it as a PyTorch DataLoader:

.. code-block:: python
    from TinyShakespeareLoader.hamlet import get_data


    data = get_data()

    train_dataloader, test_dataloader = data["train_dataloader"], data["test_dataloader"]

    for batch in train_dataloader:
        print(batch)






.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.4. For details and usage
information on PyScaffold see https://pyscaffold.org/.
