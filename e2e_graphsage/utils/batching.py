"""
Script containing helper functions to split list into chunks/batches
"""

from __future__ import absolute_import
from __future__ import division

import math


class BufferExceedError(Exception):
    """ Raised when buffer will unavoidably be exceeded """
    pass


def chunk(iterable, n):
    """Splits a list into n equal parts"""
    iterable = [e for e in iterable]
    avg_length = int(math.ceil(len(iterable) / n))
    return [iterable[i * avg_length:(i + 1) * avg_length] for i in range(n)]


def batch(iterable, n):
    """ Splits a list into batches of size n """
    iterable = [e for e in iterable]
    size = len(iterable)
    return [iterable[i:i + n] for i in range(0, size, n)]


def shingle(iterable, n):
    """
    Shingle a list into tokens of length n
    Eg. with n = 3
        [1, 2, 3, 4, 5] => [[1,2,3], [2,3,4], [3,4,5]]
    """
    num_shingles = max(1, len(iterable) - n + 1)
    return [iterable[i:i + n] for i in range(num_shingles)]


def split_cond(f, iterable):
    """
    Splits a list based on a condition
    Eg. with f = lambda x: x < 5
        [1, 9, 9, 1, 9, 1] => [[1, 9, 9], [1, 9], [1]]
        [1] => [[1]]
        [9] => []
    """
    split_point = [i for i, e in enumerate(iterable) if f(e)]
    split_point += [len(iterable)]
    return [iterable[i:j] for i, j in zip(split_point[:-1], split_point[1:])]


def batch_by_size(iterable, max_buffer=20000):
    """
    Batches an iterable by character size, useful for sending large restful
    requests
    """
    all_batches = []
    current_batch = []
    current_size = 0

    for next_item in iterable:
        # An approximated way to determine size
        next_size = len(str(next_item))
        expected_total_size = current_size + next_size

        if next_size > max_buffer:
            raise BufferExceedError('Buffer exceeded')

        elif expected_total_size > max_buffer:
            # If expected to exceed max size, then current batch is finalized
            all_batches.append(current_batch)
            current_batch = [next_item]
            current_size = next_size

        else:
            # Else add current set of instructions to current batch
            current_batch.append(next_item)
            current_size = expected_total_size

    # Group remaining instructions as a single batch
    if len(current_batch) > 0:
        all_batches.append(current_batch)

    return all_batches
