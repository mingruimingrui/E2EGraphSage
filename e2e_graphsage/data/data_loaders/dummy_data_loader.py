import torch.utils.data

default_collate = torch.utils.data._utils.collate.default_collate


class DummyDataLoader:
    """
    This is a dummy class to be used as a drop in for torch dataloader
    Used inplace of the standard torch.utils.data.DataLoader because of the
    high overhead cost of the messenger queue.
    It is highly recommended to set query_dataset with list to True
    """
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        collate_fn=default_collate,
        drop_last=False,
        query_dataset=True
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.drop_last = drop_last

        if batch_sampler is not None:
            if batch_size > 1 or shuffle or sampler is not None or drop_last:
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
                    sampler = torch.utils.data.RandomSampler(dataset)
                else:
                    sampler = torch.utils.data.SequentialSampler(dataset)
            batch_sampler = torch.utils.data.BatchSampler(
                sampler, batch_size, drop_last)

        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.query_dataset = bool(query_dataset)

    def __iter__(self):
        if self.query_dataset:
            for b in self.batch_sampler:
                new_batch = self.dataset[b]
                yield self.collate_fn(new_batch)
        else:
            for b in self.batch_sampler:
                new_batch = [self.dataset[i] for i in b]
                yield self.collate_fn(new_batch)

    def __len__(self):
        return len(self.batch_sampler)
