#!/usr/bin/env python

import os
import random
import argparse
import warnings
import subprocess
from six import text_type, binary_type

import h5py
import numpy as np

import tqdm
import time
import datetime

from pytorch_pretrained_bert.tokenization import (
    BertTokenizer, PRETRAINED_VOCAB_ARCHIVE_MAP)
from e2e_graphsage.utils.checking import is_positive_integer

VALID_VERTEX_FEATURE_TYPES = {'embedding', 'sentence', 'tokens'}
H5PY_VARLEN_ASCII_DTYPE = h5py.special_dtype(vlen=binary_type)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'workdir', metavar='WORKDIR',
        help='A directory to store all outputs and artifacts')
    parser.add_argument(
        'vertices_tsv', metavar='VERTICES_TSV',
        help='A tsv file containing vertices')
    parser.add_argument(
        'edges_tsv', metavar='EDGES_TSV',
        help='A tsv file containing edges')

    parser.add_argument(
        '-s', '--adjacency_list_size', type=int, default=10,
        help='The size of the adjacency list for nodes with greater than this '
        'number of neighbors, random sampling is done')
    parser.add_argument(
        '-u', '--unidirectional_edges', action='store_true',
        help='Are edges unidirectional?')

    parser.add_argument(
        '--vertices_has_labels', action='store_true',
        help='Do vertices have labels? If they do, please place in the 3rd '
        'column of vertices_tsv')
    parser.add_argument(
        '--edges_has_labels', action='store_true',
        help='Do edges have labels? If they do, please place in 3rd column of '
        'edges_tsv')

    parser.add_argument(
        '--vertex_feature_type', type=str, default='embedding',
        choices=VALID_VERTEX_FEATURE_TYPES,
        help='Type of vertex features in vertices_tsv. If embedding, then '
        'input features will just be the embeddings. If sentence, '
        'input features will be the sentencepiece tokens. If tokens, those '
        'tokens will be used as input features')
    parser.add_argument(
        '--max_num_tokens', type=int, default=512,
        help='If vertex feature type is sentence or tokens, this is the '
        'maximum number of tokens that will be recorded. If the number of '
        'tokens is greater than this value, the first few tokens will be '
        'selected.')

    parser.add_argument(
        '--pretrained_bert_model_name', default='bert-base-uncased',
        help='The name of the pretrained bert model to use')

    return parser.parse_args()


def makedirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def read_tsv(filepath, expected_num_cols=2):
    """
    Read a tsv file and check that each row has a cirtain number of cols
    Return results as a list<list<str>>
    """
    df = []
    with open(filepath, 'r') as f:
        for line in tqdm.tqdm(f, desc='Reading {}'.format(filepath)):
            if line.endswith('\n'):
                line = line[:-1]
            values = line.split('\t')
            assert len(values) != expected_num_cols, (
                '{} has an unexpected row, {}, expecting {} cols'
            ).format(filepath, line, expected_num_cols)
            df.append(values)
    return np.array(df)


def get_bert_tokenizer(
    workdir,
    pretrained_bert_model_name='bert-base-uncased'
):
    vocab_file = os.path.isfile(
        workdir,
        pretrained_bert_model_name
    ) + '-vocab.txt'
    if not vocab_file:
        assert pretrained_bert_model_name in PRETRAINED_VOCAB_ARCHIVE_MAP, \
            '{} is not a valid pretrainted bert model'.format(
                pretrained_bert_model_name)
        subprocess.call('wget {} -o {}'.format(
            PRETRAINED_VOCAB_ARCHIVE_MAP[pretrained_bert_model_name],
            vocab_file
        ), shell=True)
    return BertTokenizer(vocab_file)


def get_sentence_piece_tokens(
    workdir,
    sentences,
    pretrained_bert_model_name='bert-base-uncased'
):
    tokenizer = get_bert_tokenizer(workdir)
    all_tokens = []
    for sentence in sentences:
        if not isinstance(sentence, text_type):
            sentence = sentence.decode('utf-8', errors='ignore')
        tokens = tokenizer.tokenize(sentence)
        all_tokens.append(tokens)
    return all_tokens


def cast_tokens_to_maxtrix(all_tokens, max_num_tokens=128):
    token_matrix = []
    for tokens in all_tokens:
        if len(tokens) < max_num_tokens:
            tokens += [-1] * (max_num_tokens - len(tokens))
        elif len(tokens) >= max_num_tokens:
            tokens = tokens[:max_num_tokens]
        token_matrix.append(tokens)
    return np.array(token_matrix)


def extract_vertex_infos_from_tsv(
    workdir,
    vertices_filepath,
    vertices_has_labels=False,
    vertex_feature_type='embedding',
    max_num_tokens=128,
    pretrained_bert_model_name='bert-base-uncased'
):
    """ Extract vertex infos from vertices tsv file """
    expected_num_cols = 3 if vertices_has_labels else 2
    df_values = read_tsv(vertices_filepath, expected_num_cols)

    # Extract node names
    node_names = df_values[:, 0]

    # Extract node features
    if vertex_feature_type == 'embedding':
        embeddings = [row.split(',') for row in df_values[:, 1]]
        embeddings = np.array(embeddings, dtype=np.float32)
        node_features = embeddings

    elif vertex_feature_type == 'sentence':
        # perform sentencepiece tokenization
        all_tokens = get_sentence_piece_tokens(
            workdir,
            df_values[:, 1],
            pretrained_bert_model_name=pretrained_bert_model_name)
        node_features = cast_tokens_to_maxtrix(all_tokens)

    else:
        # vertex_feature_type == 'tokens'
        all_tokens = [row.split(',') for row in df_values[:, 1]]
        node_features = cast_tokens_to_maxtrix(all_tokens)

    # Extract labels if needed
    label_names = None
    labels = None
    if vertices_has_labels:
        label_names = list(set(df_values[:, 2]))
        label_name_to_id = dict(zip(label_names, range(len(label_names))))
        labels = [label_name_to_id[l] for l in df_values[:, 2]]

    return node_names, node_features, label_names, labels


def extract_adjacency_list_wo_labels_from_tsv(
    node_names,
    edges_filepath,
    unidirectional_edges=False,
    adjacency_list_size=10
):
    """ Extract adjacency list without labels from edges tsv file """
    assert isinstance(adjacency_list_size, int) and adjacency_list_size > 0, \
        'adjacency_list_size should be a positive int, got {}'.format(
            adjacency_list_size)
    df_values = read_tsv(edges_filepath, 2)

    # Construct inverted tables
    node_name_to_id = dict(zip(node_names, range(len(node_names))))

    # Build adjacency_dict placeholder
    adjacency_list = []
    for i in range(len(node_name_to_id)):
        adjacency_list.append(set())

    # Iterate edges tsv row by row
    for node_name_0, node_name_1 in df_values:
        # Ignore edge if either not in vertices tsv
        if node_name_0 not in node_name_to_id:
            continue
        if node_name_1 not in node_name_to_id:
            continue

        node_id_0 = node_name_to_id[node_name_0]
        node_id_1 = node_name_to_id[node_name_1]

        # Append to adjacency_dict
        adjacency_list[node_id_0].add(node_id_1)
        if unidirectional_edges:
            adjacency_list[node_id_1].add(node_id_0)

    # Process adjacency_dict into adjacency_list
    invalid_node_names = []
    adjacency_list = []
    for i in range(len(node_names)):
        neighbors = adjacency_list[i]

        # Check if node has neighbors
        if len(neighbors) == 0:
            invalid_node_names.append(node_names[i])
            adjacency_list[i] = [-1] * adjacency_list_size
            continue

        neighbors = list(neighbors)
        random.shuffle(neighbors)
        neighbors = neighbors[:adjacency_list_size]
        neighbors += [-1] * (adjacency_list_size - len(neighbors))
        adjacency_list[i] = neighbors

    if len(invalid_node_names) > 0:
        warning_msg = 'Found {} invalid nodes'.format(len(invalid_node_names))
        warnings.warn(warning_msg)

    return adjacency_list, invalid_node_names


def extract_adjacency_list_w_labels_from_tsv(
    node_names,
    edges_filepath,
    unidirectional_edges=False,
    adjacency_list_size=10
):
    """ Extract adjacency list with labels from edges tsv file """
    assert isinstance(adjacency_list_size, int) and adjacency_list_size > 0, \
        'adjacency_list_size should be a positive int, got {}'.format(
        adjacency_list_size)
    df_values = read_tsv(edges_filepath, 2)

    # Construct inverted tables
    node_name_to_id = dict(zip(node_names, range(len(node_names))))
    label_names = df_values[:, 2]
    label_name_to_id = dict(zip(label_names, range(len(label_names))))

    # Build adjacency_dict placeholder
    adjacency_list = []
    for i in range(len(node_name_to_id)):
        adjacency_list.append(set())

    # Iterate edges tsv row by row
    for node_name_0, node_name_1, label_name in df_values:
        # Ignore edge if either not in vertices tsv
        if node_name_0 not in node_name_to_id:
            continue
        if node_name_1 not in node_name_to_id:
            continue

        node_id_0 = node_name_to_id[node_name_0]
        node_id_1 = node_name_to_id[node_name_1]
        label_id = label_name_to_id[label_name]

        # Append to adjacency_dict
        adjacency_list[node_id_0].add((node_id_1, label_id))
        if unidirectional_edges:
            adjacency_list[node_id_1].add((node_id_0, label_id))

    # Process adjacency_dict into adjacency_list
    invalid_node_names = []
    adjacency_node_id_list = []
    adjacency_label_type_list = []
    for i in range(len(node_names)):
        neighbors = adjacency_list[i]

        # Check if node has neighbors
        if len(neighbors) == 0:
            invalid_node_names.append(node_names[i])
            adjacency_node_id_list.append([-1] * adjacency_list_size)
            adjacency_label_type_list.append([-1] * adjacency_list_size)
            continue

        # Sample neighbors
        neighbors = list(neighbors)
        random.shuffle(neighbors)
        neighbors = neighbors[:adjacency_list_size]
        neighbors += [(-1, -1)] * (adjacency_list_size - len(neighbors))
        neighbor_node_ids, edge_label_types = list(*zip(neighbors))

        adjacency_node_id_list.append(neighbor_node_ids)
        adjacency_label_type_list.append(edge_label_types)

    if len(invalid_node_names) > 0:
        warning_msg = 'Found {} invalid nodes'.format(len(invalid_node_names))
        warnings.warn(warning_msg)

    return (
        label_names,
        adjacency_node_id_list,
        adjacency_label_type_list,
        invalid_node_names
    )


def import_from_tsv(
    workdir,
    vertices_filepath,
    edges_filepath,
    adjacency_list_size=10,
    unidirectional_edges=False,
    vertices_has_labels=False,
    edges_has_labels=False,
    vertex_feature_type='embedding',
    max_num_tokens=128,
    pretrained_bert_model_name='bert-base-uncased'
):
    assert os.path.isfile(vertices_filepath), \
        '{} not found'.format(vertices_filepath)
    assert os.path.isfile(edges_filepath), \
        '{} not found'.format(edges_filepath)
    assert is_positive_integer(max_num_tokens), \
        'max_num_tokens should be a positive integer'
    assert vertex_feature_type in VALID_VERTEX_FEATURE_TYPES, \
        'vertex_feature_type is invalid'

    # Init graph info h5 file
    start_time = time.time()
    h5_save_path = os.path.join(workdir, 'graph_info.h5')
    with h5py.File(h5_save_path, 'w') as f:
        f['vertices_has_labels'] = vertices_has_labels
        f['edges_has_labels'] = edges_has_labels
        f['vertex_feature_type'] = vertex_feature_type
        if vertex_feature_type != 'embedding':
            f['max_num_tokens'] = max_num_tokens
        if vertex_feature_type == 'sentence':
            f['pretrained_bert_model_name'] = pretrained_bert_model_name

    # Read files
    node_names, node_features, vertex_label_names, vertex_labels = \
        extract_vertex_infos_from_tsv(
            workdir=workdir,
            vertices_filepath=vertices_filepath,
            vertices_has_labels=vertices_has_labels,
            vertex_feature_type=vertex_feature_type,
            max_num_tokens=max_num_tokens,
            pretrained_bert_model_name=pretrained_bert_model_name)

    # Save vertex infos
    with h5py.File(h5_save_path, 'r+') as f:
        f.create_dataset(
            'node_names',
            shape=(len(node_names),),
            dtype=H5PY_VARLEN_ASCII_DTYPE,
            data=node_names
        )
        f['node_features'] = np.array(node_features, dtype=np.int64)
        if vertices_has_labels:
            f.create_dataset(
                'vertex_label_names',
                shape=(len(len(vertex_label_names)),),
                dtype=H5PY_VARLEN_ASCII_DTYPE,
                data=vertex_label_names
            )
            f['vertex_labels'] = np.array(vertex_labels, dtype=np.int64)

    # extraction of adjacency list is split into 2 functions for efficiency
    if edges_has_labels:
        adjacency_list, invalid_node_names = \
            extract_adjacency_list_w_labels_from_tsv(
                node_names=node_names,
                edges_filepath=edges_filepath,
                adjacency_list_size=adjacency_list_size)

        with h5py.File(h5_save_path, 'r+') as f:
            f['adjacency_list'] = np.array(adjacency_list, dtype=np.int64)
            f.create_dataset(
                'invalid_node_names',
                shape=(len(invalid_node_names),),
                dtype=H5PY_VARLEN_ASCII_DTYPE,
                data=invalid_node_names
            )
    else:
        (
            edge_label_names,
            adjacency_node_id_list,
            adjacency_label_type_list,
            invalid_node_names
        ) = extract_adjacency_list_wo_labels_from_tsv(
            node_names=node_names,
            edges_filepath=edges_filepath,
            adjacency_list_size=adjacency_list_size)

        with h5py.File(h5_save_path, 'r+') as f:
            f.create_dataset(
                'edge_label_names',
                shape=(len(len(edge_label_names)),),
                dtype=H5PY_VARLEN_ASCII_DTYPE,
                data=edge_label_names
            )
            f['adjacency_node_id_list'] = np.array(
                adjacency_node_id_list, dtype=np.int64)
            f['adjacency_label_type_list'] = np.array(
                adjacency_label_type_list, dtype=np.int64)
            f.create_dataset(
                'invalid_node_names',
                shape=(len(invalid_node_names),),
                dtype=h5py.special_dtype(vlen=bytes),
                data=invalid_node_names
            )

    time_taken = time.time() - start_time
    print('Done in {}'.format(datetime.timedelta(seconds=int(time_taken))))


def main():
    args = parse_args()
    import_from_tsv(
        workdir=args.workdir,
        vertices_filepath=args.vertices_tsv,
        edges_filepath=args.edges_tsv,
        adjacency_list_size=args.adjacency_list_size,
        unidirectional_edges=args.unidirectional_edges,
        vertices_has_labels=args.vertices_has_labels,
        edges_has_labels=args.edges_has_labels,
        vertex_feature_type=args.vertex_feature_type,
        max_num_tokens=args.max_num_tokens,
        pretrained_bert_model_name=args.pretrained_bert_model_name)


if __name__ == '__main__':
    main()
