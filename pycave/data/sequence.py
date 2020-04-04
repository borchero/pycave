from torch.nn.utils.rnn import pack_sequence
from torch.utils.data import Dataset

class SequenceDataset(Dataset):
    """
    SequenceDataset models a set of sequences. It exposes a collation function that packs the
    sequences to obtain batches.
    """

    @classmethod
    def collate_fn(cls, items):
        """
        Collates the items yielded by this dataset into a single batch by packing the sequences.

        Parameters
        ----------
        items: list of torch.Tensor
            Sequences of possibly different length.
        """
        return pack_sequence(items, enforce_sorted=False)

    def __init__(self, indexable):
        """
        Initializes a new sequence dataset with the given sequences.

        Parameters
        ----------
        indexable: indexable of torch.Tensor
            A type with an implementation for `__len__` and `__getitem__` where the latter returns a
            single sequence. Might e.g. be a list, a multi-dimensional tensor, or a dataset.
        """
        self.indexable = indexable

    def __len__(self):
        return len(self.indexable)

    def __getitem__(self, index):
        return self.indexable[index]
