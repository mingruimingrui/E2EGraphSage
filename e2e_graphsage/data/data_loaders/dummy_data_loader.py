from torch.utils.data import SequentialSampler, RandomSampler, BatchSampler
from torch.utils.data import _utils

default_collate = _utils.collate.default_collate


class DummyDataLoader(object):
    """
    This is a dummy class to be used as a drop in for torch dataloader
    Used inplace of the standard torch.utils.data.DataLoader because of the
    high overhead cost of the messenger queue
    """
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        collate_fn=default_collate,
        get_with_list=False
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.get_with_list = bool(get_with_list)

        if batch_sampler is not None:
            if batch_size > 1 or shuffle or sampler is not None:
                raise ValueError('batch_sampler option is mutually exclusive '
                                 'with batch_size, shuffle, sampler, and '
                                 'drop_last')
            self.batch_size = None
            self.drop_last = None

        if sampler is not None and shuffle:
            raise ValueError('sampler option is mutually exclusive with '
                             'shuffle')

        if batch_sampler is None:
            if sampler is None:
                if shuffle:
                    sampler = RandomSampler(dataset)
                else:
                    sampler = SequentialSampler(dataset)
            batch_sampler = BatchSampler(sampler, batch_size)

        self.sampler = sampler
        self.batch_sampler = batch_sampler

    def __len__(self):
        return len(self.batch_sampler)

    def __iter__(self):
        if self.get_with_list:
            for b in self.batch_sampler:
                yield self.collate_fn(self.dataset[b])
        else:
            for b in self.batch_sampler:
                yield self.collate_fn([self.dataset[i] for i in b])
