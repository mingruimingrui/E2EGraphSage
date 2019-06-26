#!/usr/bin/env python
from __future__ import division

import os
import random
import warnings
import argparse

import snap
import h5py
import numpy as np
from six import binary_type

import tqdm
import time
import datetime


SCRIPT_DESCRIPTION = "Parse tsv files into input format"

SAVE_PREFIX_HELP_MSG = """
The prefix that will be used to save objects with. One can also
control save path with this argument.

A total of 2 file will be produced, <SAVE_PREFIX>.graph and <SAVE_PREFIX>.h5
"""

VERTICES_TSV_HELP_MSG = """
This should be a header-less tab separated file containing 2 columns (or 3 if
vertices_contain_labels). Corollary, it is expected that each line of this file
contains exactly 1 tab (or 2 if vertices_contain_labels).

- First column should contain node names

- Second column should contain the node features which can be one of 3 format
  this format is selected using vertices_data_format
    - embeddings: a comma separated string containng the embedding vector, must
                 be of fixed len
    - tokens: a comma separated string containing token ids, can be variable
              len but expects tokens to be non negative integers and contiguous
    - sentence: a text sentence

- Third column (if vertices_contain_labels) should contain the label_id
  label_ids are expected to be contiguous non-negative integers
"""

EDGES_CSV_HELP_MSG = """
This should be a header-less tab separated file containing 2 columns

- First column should the source node name
- Second column should the destination node name
"""

H5PY_VARLEN_ASCII_DTYPE = h5py.special_dtype(vlen=binary_type)
VALID_VERTEX_DATA_FORMATS = {'embeddings', 'tokens', 'sentence'}


def parse_args():
    parser = argparse.ArgumentParser(description=SCRIPT_DESCRIPTION)

    parser.add_argument(
        'save_prefix', metavar='SAVE_PREFIX', type=str,
        help=SAVE_PREFIX_HELP_MSG)
    parser.add_argument(
        'vertices_tsv', metavar='VERTICES_TSV', type=str,
        help=VERTICES_TSV_HELP_MSG)
    parser.add_argument(
        'edges_tsv', metavar='EDGES_TSV', type=str,
        help=EDGES_CSV_HELP_MSG)

    parser.add_argument(
        '-vf', '--vertices_data_format', type=str, default='embeddings',
        choices=VALID_VERTEX_DATA_FORMATS,
        help='Which format is expected of vertices_tsv? '
        'Should be one of {}'.format(VALID_VERTEX_DATA_FORMATS)
    )
    parser.add_argument(
        '-vl', '--vertices_contain_labels', action='store_true',
        help='Does vertices contain labels?')
    parser.add_argument(
        '-u', '--unidirectional_edges', action='store_true',
        help='Are edges unidirectional?')

    parser.add_argument(
        '-n', '--num_samples', type=int, default=10,
        help='How many neighbors should each node sample?')
    parser.add_argument(
        '-nt', '--max_num_tokens', type=int, default=128,
        help='The maximum number of tokens for each node, only applicable for '
        'vertices_dataformat in [tokens, sentence]')

    parser.add_argument(
        '--num_workers', type=int, default=0,
        help='The number of users to use to process input features')

    return parser.parse_args()


def extract_vertex_infos_from_tsv(
    vertices_tsv_filepath,
    vertices_data_format='embeddings',
    vertices_contain_labels=False,
    max_num_tokens=128,
    num_workers=0
):
    """
    Extracts node_names, node_features and vertex_labels from vertices_tsv
    """
    expected_num_cols = 3 if vertices_contain_labels else 2
    node_names = []
    node_features = []
    vertex_labels = [] if vertices_contain_labels else None

    def parse_line(line):
        if line.endswith(b'\n'):
            line = line[:-1]
        row = line.split(b'\t')
        assert len(row) == expected_num_cols, (
            'Found row in vertices_tsv that does not match expected '
            'number of columns, expected {} columns, got {}. raw line: {}'
        ).format(expected_num_cols, len(row), line)

        node_names.append(row[0])
        node_features.append(row[1])
        vertex_labels.append(row[2])

    # Read vertices_tsv line by line
    with open(vertices_tsv_filepath, 'rb') as f:
        for line in tqdm.tqdm(f, desc='Reading vertices_tsv', ncols=80):
            parse_line(line)

    # Ensure that node_names are unique
    assert len(node_names) == len(set(node_names)), \
        'There are repeated nodes in vertices_tsv'

    def cast_to_binary(text):
        if isinstance(text, binary_type):
            binary = text
        else:
            binary = text.encode('utf-8', errors='ignore')
        return binary

    node_names = list(map(cast_to_binary, node_names))
    node_names = np.array(node_names)

    # Parse node_features
    # todo: use multiprocessing for this part
    if vertices_data_format == 'embeddings':
        def parse_embeddings(feature_str):
            return list(map(float, feature_str.split(',')))
        node_embeddings = []
        desc = 'Parsing embeddings'
        for feature_str in tqdm.tqdm(node_features, desc=desc, ncols=80):
            node_embeddings.append(parse_embeddings(feature_str))
        node_features = np.array(node_embeddings, dtype=np.float32)

    elif vertices_data_format == 'tokens':
        raise NotImplementedError()

    elif vertices_data_format == 'sentence':
        raise NotImplementedError()

    else:
        raise ValueError('{} is not a valid vertices_data_format'.format(
            vertices_data_format))

    if vertices_contain_labels:
        # Parse vertex_labels
        vertex_labels = list(map(int, vertex_labels))
        assert all(map(lambda x: x >= 0, vertex_labels)), \
            'vertex_labels are expected to be non negative integers'
        num_classes = max(vertex_labels) + 1
        unique_given_vertex_labels = set(vertex_labels)
        for i in range(num_classes):
            if i not in unique_given_vertex_labels:
                warning_msg = 'No examples for class {} is given'.format(i)
                warnings.warn(warning_msg)
        vertex_labels = np.array(vertex_labels, dtype=np.int64)

    return node_names, node_features, vertex_labels


def extract_adjacency_list_from_tsv(
    node_names,
    edges_tsv_filepath,
    unidirectional_edges=False,
    num_samples=10
):
    """  """
    node_names = [n.decode('utf-8') for n in node_names]
    node_name_to_idx = dict(zip(node_names, range(len(node_names))))
    if unidirectional_edges:
        G = snap.TNGraph.New()
    else:
        G = snap.TUNGraph.New()
    for node_name in node_names:
        G.AddNode(node_name_to_idx[node_name])

    def parse_line(line):
        if line.endswith(b'\n'):
            line = line[:-1]
        row = line.split(b'\t')
        assert len(row) == 2, (
            'Found row in edges_tsv that does not match expected '
            'number of columns, expected {} columns, got {}. raw line: {}'
        ).format(2, len(row), line)

        src_node_name = row[0]
        dst_node_name = row[1]

        if src_node_name not in node_name_to_idx:
            return
        if dst_node_name not in node_name_to_idx:
            return

        G.AddEdge(
            node_name_to_idx[src_node_name],
            node_name_to_idx[dst_node_name]
        )

    # Read edges_tsv line by line
    num_edges = 0
    with open(edges_tsv_filepath, 'rb') as f:
        for _ in tqdm.tqdm(f, desc='Counting edges_tsv lines', ncols=80):
            num_edges += 1
    with open(edges_tsv_filepath, 'rb') as f:
        pbar = tqdm.tqdm(
            f,
            total=num_edges,
            desc='Reading edges_tsv',
            ncols=80
        )
        for line in pbar:
            parse_line(line)
    G.Defrag()
    print('Created graph with {} nodes and {} edges'.format(
        G.GetNodes(), G.GetEdges()))

    # Make adjacency_list
    adjacency_list = []
    desc = 'Making adjacency list'
    for node_name in tqdm.tqdm(node_names, desc=desc, ncols=80):
        src_node = G.GetNI(node_name_to_idx[node_name])
        num_neighbors = src_node.GetDeg()
        neighbors = [src_node.GetNbrNId(i) for i in range(num_neighbors)]
        random.shuffle(neighbors)
        if num_neighbors > num_samples:
            neighbors = neighbors[:num_samples]
        elif num_neighbors < num_samples:
            neighbors += [-1] * (num_samples - len(neighbors))
        adjacency_list.append(neighbors)
    adjacency_list = np.array(adjacency_list, dtype=np.int64)

    return G, adjacency_list


def import_from_tsv(
    save_prefix,
    vertices_tsv_filepath,
    edges_tsv_filepath,
    vertices_data_format='embeddings',
    vertices_contain_labels=False,
    unidirectional_edges=False,
    num_samples=10,
    max_num_tokens=128,
    num_workers=0
):
    """
    Import graph data from tsv files
    """
    start_time = time.time()

    assert os.path.isfile(vertices_tsv_filepath), \
        '{} not found'.format(vertices_tsv_filepath)
    assert os.path.isfile(edges_tsv_filepath), \
        '{} not found'.format(edges_tsv_filepath)
    assert vertices_data_format in VALID_VERTEX_DATA_FORMATS, \
        'vertices_data_format is invalid, got {}'.format(vertices_data_format)
    assert isinstance(num_samples, int) and num_samples > 0, \
        'num_samples should be a positive integer'
    assert isinstance(max_num_tokens, int) and max_num_tokens > 0, \
        'max_num_tokens should be a positive integer'

    # Make savedir
    dirname = os.path.dirname(save_prefix)
    if dirname != '' and not os.path.isdir(dirname):
        os.makedirs(dirname)
    snap_filepath = save_prefix + '.graph'
    h5_filepath = save_prefix + '.h5'

    # Init h5 file
    with h5py.File(h5_filepath, 'w') as f:
        f['vertices_contain_labels'] = vertices_contain_labels
        f['vertices_data_format'] = vertices_data_format
        if vertices_data_format != 'embeddings':
            f['max_num_tokens'] = max_num_tokens

    # Read vertices_tsv
    node_names, node_features, vertex_labels = extract_vertex_infos_from_tsv(
        vertices_tsv_filepath=vertices_tsv_filepath,
        vertices_data_format=vertices_data_format,
        vertices_contain_labels=vertices_contain_labels,
        max_num_tokens=max_num_tokens,
        num_workers=num_workers
    )
    with h5py.File(h5_filepath, 'r+') as f:
        f.create_dataset(
            'node_names',
            shape=(len(node_names),),
            dtype=H5PY_VARLEN_ASCII_DTYPE,
            data=node_names
        )
        f['node_features'] = node_features
        if vertices_contain_labels:
            f['vertex_labels'] = vertex_labels
    del node_features, vertex_labels

    # Read edges_tsv
    G, adjacency_list = extract_adjacency_list_from_tsv(
        node_names=node_names,
        edges_tsv_filepath=edges_tsv_filepath,
        unidirectional_edges=unidirectional_edges,
        num_samples=num_samples
    )
    FOut = snap.TFOut(snap_filepath)
    G.Save(FOut)
    FOut.Flush()
    with h5py.File(h5_filepath, 'r+') as f:
        f['adjacency_list'] = adjacency_list

    seconds_taken = int(time.time() - start_time)
    print('Import completed in {}'.format(
        datetime.timedelta(seconds=seconds_taken)))


def main():
    args = parse_args()
    import_from_tsv(
        save_prefix=args.save_prefix,
        vertices_tsv_filepath=args.vertices_tsv,
        edges_tsv_filepath=args.edges_tsv,
        vertices_data_format=args.vertices_data_format,
        vertices_contain_labels=args.vertices_contain_labels,
        unidirectional_edges=args.unidirectional_edges,
        num_samples=args.num_samples,
        max_num_tokens=args.max_num_tokens,
        num_workers=args.num_workers
    )


if __name__ == '__main__':
    main()
