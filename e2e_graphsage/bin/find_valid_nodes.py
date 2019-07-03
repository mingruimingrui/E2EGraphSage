#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script can be used to identify valid training nodes from an imported graph
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tqdm
import h5py
import random
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'save_prefix', metavar='SAVE_PREFIX', type=str,
        help='Save prefix used in import_from_csv')

    parser.add_argument(
        'output_file', metavar='OUTPUT', type=str,
        help='valid node names will be output to this file')

    parser.add_argument(
        '-d', '--depth', type=int, default=1,
        help='The depth used in bfs')
    parser.add_argument(
        '-k', '--cutoff', type=int, default=1,
        help='The minimum number of neighbors that each node should contain.')

    parser.add_argument(
        '--train_val_split', type=float, default=1,
        help='If provided, this much percent will be split into training '
        'set. The remaining will go into validation set. Must be a value '
        'between 0 and 1.')
    parser.add_argument(
        '--num_val', type=int, default=None,
        help='An alternative to train_val_split. If provided, then this number'
        'of vertices will be selected as the validation set')

    return parser.parse_args()


def find_valid_nodes(
    save_prefix,
    depth=1,
    cutoff=1
):
    assert isinstance(depth, int) and depth > 0, \
        'depth should be a non negative integer'
    assert isinstance(cutoff, int) and cutoff > 0, \
        'cutoff should be a non negative integer'

    # Read node info from h5
    with h5py.File(save_prefix + '.h5', 'r') as f:
        adjacency_list = f['adjacency_list'][:]

    # Find valid node ids
    def _find_valid_nodeids(valid_nodeids, adjacency_list):
        new_valid_nodeids = set()
        desc = 'Finding valid nodes'
        for nodeid in tqdm.tqdm(valid_nodeids, desc=desc, ncols=80):
            num_valid_neighbors = 0
            row = adjacency_list[nodeid]
            for neighbor_id in row[row != -1]:
                if neighbor_id in valid_nodeids:
                    num_valid_neighbors += 1
            if num_valid_neighbors >= cutoff:
                new_valid_nodeids.add(nodeid)
        return new_valid_nodeids

    valid_nodeids = set(range(len(adjacency_list)))
    for i in range(depth):
        print('Starting to find at depth - {}'.format(i))
        valid_nodeids = _find_valid_nodeids(
            valid_nodeids,
            adjacency_list
        )
    valid_nodeids = list(valid_nodeids)
    valid_nodeids.sort()

    return valid_nodeids


def write_node_names_to_file(node_names, filename):
    with open(filename, 'w') as f:
        for node_name in node_names:
            f.write(node_name + '\n')


def main():
    args = parse_args()
    valid_nodeids = find_valid_nodes(
        save_prefix=args.save_prefix,
        depth=args.depth,
        cutoff=args.cutoff
    )
    print('Found {} valid nodes'.format(len(valid_nodeids)))

    with h5py.File(args.save_prefix + '.h5', 'r') as f:
        node_names = f['node_names'][:]
    valid_node_names = node_names[valid_nodeids]

    print('Writing valid node ids to file')
    if args.num_val is not None:
        assert isinstance(args.num_val, int) and args.num_val > 0, \
            'num_val should be a non negative integer'

        random.shuffle(valid_node_names)
        train_nodeids = valid_node_names[args.num_val:]
        val_nodeids = valid_node_names[:args.num_val]

        write_node_names_to_file(train_nodeids, args.output_file + '.train')
        write_node_names_to_file(val_nodeids, args.output_file + '.val')

    elif args.train_val_split == 1:
        write_node_names_to_file(valid_node_names, args.output_file)

    else:
        assert 0 < args.train_val_split <= 1, \
            'train_val_split should be a value between 0 and 1'

        num_train = len(valid_node_names) * float(args.train_val_split)
        num_train = int(num_train)

        random.shuffle(valid_node_names)
        train_nodeids = valid_node_names[:num_train]
        val_nodeids = valid_node_names[num_train:]

        write_node_names_to_file(train_nodeids, args.output_file + '.train')
        write_node_names_to_file(val_nodeids, args.output_file + '.val')


if __name__ == "__main__":
    main()
