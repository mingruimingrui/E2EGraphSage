from __future__ import division

import math
import random
import warnings

import torch.utils.data

from ...utils.batching import batch as batch_fn


class NegativeBatchSampler(torch.utils.data.BatchSampler):
    """
    Batch sampler used to sample for negative and hard negatives

    Take note, if no hard negative sampling is done, then this turns into
    random sampling
    """
    def __init__(
        self,
        dataset_len,
        num_iters,
        batch_size,
        labels=None,
        num_hard_negative_samples=0
    ):
        """
        Args:
            dataset_len (int): The size of the torch.utils.data.dataset to
                sample on
            num_iters (int): The number of batches to iterate, this
                value will be used for pre-computation of items to place in
                each batch
            batch_size (int): The batch size
            labels (list<int>): A list of labels, labels just have to be
                hashable, not needed if no hard_negative_sampling is needed
            num_hard_negative_samples: The number of hard negative samples
                to sample for each example
        """
        assert isinstance(dataset_len, int) and dataset_len > 0, \
            'dataset_len should be a positive number'
        assert isinstance(num_iters, int) and num_iters > 0, \
            'num_iters should be a positive number'
        assert isinstance(batch_size, int) and batch_size > 0, \
            'batch_size should be a positive number'
        assert isinstance(num_hard_negative_samples, int) and \
            num_hard_negative_samples >= 0, \
            'num_hard_negative_samples should be a non negative number'

        if num_hard_negative_samples > 0:
            assert labels is not None, \
                'labels has to be provided to do hard negative sampling'
            assert len(labels) == dataset_len, \
                'Number of labels mismatch with dataset_len'

            self.labels = labels
            self.unique_labels = set(self.labels)
            self.unique_label_to_idx = {l: [] for l in self.unique_labels}
            for i, l in enumerate(self.labels):
                self.unique_label_to_idx[l].append(i)

            for l, idx_set in self.unique_label_to_idx.items():
                if len(idx_set) < num_hard_negative_samples + 1:
                    warn_msg = (
                        'label {} has less number of entries than num_hard '
                        'num_hard_negative_samples + 1'
                    ).format(l)
                    warnings.warn(warn_msg)

            minibatch_size = math.ceil(batch_size / num_hard_negative_samples)
            minibatch_size = int(minibatch_size)

        else:
            minibatch_size = batch_size

        self.dataset_len = dataset_len
        self.num_iters = num_iters
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.num_hard_negative_samples = num_hard_negative_samples

    def __len__(self):
        return self.num_iters

    def __iter__(self):
        # Populate groups
        order = []
        while len(order) < self.num_iters * self.minibatch_size:
            _order = list(range(self.dataset_len))
            random.shuffle(_order)
            order += _order
        groups = batch_fn(order, self.minibatch_size)
        groups = groups[:self.num_iters]
        del order, _order

        # Yield batches of idx
        if self.num_hard_negative_samples == 0:
            for batch in groups:
                yield batch

        else:
            num_to_sample = self.num_hard_negative_samples + 1
            for minibatch in groups:
                batch = []
                for idx in minibatch:
                    label = self.labels[idx]

                    # Sample hard negative samples
                    all_hard_negatives = self.unique_label_to_idx[label]
                    if len(all_hard_negatives) <= num_to_sample:
                        hard_negative_samples = all_hard_negatives
                    else:
                        hard_negative_samples = random.sample(
                            all_hard_negatives, num_to_sample)

                    # Remove self
                    hard_negative_samples = \
                        [i for i in hard_negative_samples if i != idx]
                    hard_negative_samples = \
                        hard_negative_samples[:self.num_hard_negative_samples]

                    batch.append(idx)
                    batch.extend(hard_negative_samples)
                if len(batch) < self.batch_size:
                    batch += [
                        random.randint(0, self.dataset_len - 1)
                        for i in range(self.batch_size - len(batch))
                    ]
                else:
                    batch = batch[:self.batch_size]
                yield batch
