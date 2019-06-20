#!/usr/bin/env python

import os
import argparse
from six import text_type

import tqdm

import numpy as np

import sentencepiece

from pytorch_pretrained_bert.tokenization import BertTokenizer
from e2e_graphsage.utils.checking import is_positive_integer

VALID_VERTEX_FEATURE_TYPES = {'embedding', 'sentence', 'tokens'}


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
        '--max_num_tokens', type=int, default=128,
        help='If vertex feature type is sentence or tokens, this is the '
        'maximum number of tokens that will be recorded. If the number of '
        'tokens is greater than this value, the first few tokens will be '
        'selected.')

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


def get_bert_tokenizer(workdir):
    assert isinstance(workdir, 'spm_vocab.txt'), (
        'Unable to find spm_vocab.txt, please prepare some text data and do '
        'bert pretraining first'
    )


def get_sentence_piece_tokens(workdir, sentences):
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


def read_vertices_tsv(
    workdir,
    vertices_filepath,
    vertices_has_labels=False,
    vertex_feature_type='embedding',
    max_num_tokens=128
):
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
        all_tokens = get_sentence_piece_tokens(workdir, df_values[:, 1])
        node_features = cast_tokens_to_maxtrix(all_tokens)

    else:
        # vertex_feature_type == 'tokens'
        all_tokens = [row.split(',') for row in df_values[:, 1]]
        node_features = cast_tokens_to_maxtrix(all_tokens)

    # Extract labels if needed
    labels = None
    if vertices_has_labels:
        labels = df_values[:, 2]

    return node_names, node_features, labels


def import_from_tsv(
    workdir,
    vertices_filepath,
    edges_filepath,
    vertices_has_labels=False,
    edges_has_labels=False,
    vertex_feature_type='embedding',
    max_num_tokens=128
):
    assert os.path.isfile(vertices_filepath), \
        '{} not found'.format(vertices_filepath)
    assert os.path.isfile(edges_filepath), \
        '{} not found'.format(edges_filepath)
    assert is_positive_integer(max_num_tokens), \
        'max_num_tokens should be a positive integer'
    assert vertex_feature_type in VALID_VERTEX_FEATURE_TYPES, \
        'vertex_feature_type is invalid'

    # Read files


def main():
    args = parse_args()
    import_from_tsv(
        workdir=args.workdir,
        vertices_filepath=args.vertices_tsv,
        edges_filepath=args.edges_tsv,
        vertices_has_labels=args.vertices_has_labels,
        edges_has_labels=args.edges_has_labels,
        vertex_feature_type=args.vertex_feature_type,
        max_num_tokens=args.max_num_tokens
    )


if __name__ == '__main__':
    main()
