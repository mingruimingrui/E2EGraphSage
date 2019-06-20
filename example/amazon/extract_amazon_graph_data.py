#!/usr/bin/env python

"""
Extracts graph data from amazon locally

There are mainly 2 reasons for not doing this on spark
1. Not everyone has access to a spark server
2. The run time is not very long ~14 min on a pretty slow hdd
    io appears to be the main bottleneck
"""

import os
import tqdm
import atexit
import argparse
import pandas as pd
import multiprocessing

VERTICES_COLS_TO_EXTRACT = [
    'product_id',
    'product_title',
    'product_category'
]

EDGES_COLS_TO_EXTRACT = [
    'customer_id',
    'product_id',
    'star_rating',
    'helpful_votes',
    'total_votes',
    'vine',
    'verified_purchase',
    'review_date'
]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'src_dir', metavar='SRC_DIR',
        help='The directory storing all raw amazon_reviews')
    parser.add_argument(
        'dst_dir', metavar='DST_DIR',
        help='The directory used to save all amazon graph data')

    parser.add_argument(
        '-n', '--num_workers', type=int, default=0,
        help='The number of workers to use')
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Should a progress bar be shown?')

    return parser.parse_args()


def _extract_amazon_graph_data(
    filenames_queue, done_queue, coordinator,
    src_dir, dst_dir, num_workers=0, verbose=False
):
    while not coordinator['stop']:
        try:
            filename = filenames_queue.get(timeout=1)
        except Exception:
            continue

        filepath = os.path.join(src_dir, filename)
        product_category = os.path.splitext(filename)[0]
        product_category = product_category.replace('amazon_reviews_us_', '')

        vertices_df_savepath = \
            os.path.join(dst_dir, 'vertices', product_category) + '.tsv'
        edges_df_savepath = \
            os.path.join(dst_dir, 'edges', product_category) + '.tsv'

        try:
            vertices_df = []
            edges_df = []
            with open(filepath, 'r') as f:
                header_row = f.readline()[:-1].split('\t')
                vertex_col_idx_to_extract = \
                    [header_row.index(c) for c in VERTICES_COLS_TO_EXTRACT]
                edge_col_idx_to_extract = \
                    [header_row.index(c) for c in EDGES_COLS_TO_EXTRACT]
                desc = 'Extracting {}'.format(filename)
                if verbose:
                    f = tqdm.tqdm(f, desc=desc)
                else:
                    print(desc)
                for line in f:
                    row = line[:-1].split('\t')
                    vertices_df.append([row[i] for i in vertex_col_idx_to_extract])
                    edges_df.append([row[i] for i in edge_col_idx_to_extract])
        except Exception as e:
            coordinator['stop'] = True
            print('Something went wrong!')
            raise e

        vertices_df = pd.DataFrame(
            vertices_df,
            columns=VERTICES_COLS_TO_EXTRACT,
            dtype='O'
        )
        vertices_df.drop_duplicates(inplace=True)
        vertices_df.to_csv(vertices_df_savepath, sep='\t', index=False)

        edges_df = pd.DataFrame(
            edges_df,
            columns=EDGES_COLS_TO_EXTRACT,
            dtype='O'
        )
        edges_df.drop_duplicates(inplace=True)
        edges_df.to_csv(edges_df_savepath, sep='\t', index=False)

        done_queue.put(filename)


def extract_amazon_graph_data(src_dir, dst_dir, num_workers=0, verbose=False):
    all_filenames = os.listdir(src_dir)
    all_filenames = list(filter(
        lambda x: x.startswith('amazon_reviews_us_') and x.endswith('.tsv'),
        all_filenames
    ))
    assert len(all_filenames) != 0, \
        'No review files found'

    # Make dst_dir
    if not os.path.isdir(dst_dir):
        os.makedirs(dst_dir)
    if not os.path.isdir(os.path.join(dst_dir, 'vertices')):
        os.makedirs(os.path.join(dst_dir, 'vertices'))
    if not os.path.isdir(os.path.join(dst_dir, 'edges')):
        os.makedirs(os.path.join(dst_dir, 'edges'))

    # Start manager and make queues
    manager = multiprocessing.Manager()
    filenames_queue = manager.Queue()
    done_queue = manager.Queue()
    coordinator = manager.dict()
    coordinator['stop'] = False

    def request_stop():
        print('Stopping workers')
        coordinator['stop'] = True

    atexit.register(request_stop)

    # Fill filenames queue
    for filename in all_filenames:
        filenames_queue.put(filename)

    # Consume filenames queue and extract amazon products
    if num_workers > 0:
        # Run using multiple processes
        workers = []
        for _ in range(num_workers):
            p = multiprocessing.Process(
                target=_extract_amazon_graph_data,
                args=[
                    filenames_queue, done_queue, coordinator,
                    src_dir, dst_dir, num_workers, verbose
                ]
            )
            p.start()
            workers.append(p)

        num_done = 0
        while not coordinator['stop'] and num_done < len(all_filenames):
            try:
                done_queue.get(timeout=1)
                num_done += 1
            except Exception:
                continue

        if coordinator['stop']:
            raise Exception('A worker stopped unexpectedly')

        coordinator['stop'] = True
        for p in workers:
            p.join()

    elif num_workers == 0:
        # Run on main process
        for filename in all_filenames:
            _extract_amazon_graph_data(
                filenames_queue, done_queue, coordinator,
                src_dir, dst_dir, num_workers, verbose)

    else:
        # num_workers < 0
        raise ValueError('num_workers should not be a negative value')


def main():
    args = parse_args()
    extract_amazon_graph_data(
        src_dir=args.src_dir,
        dst_dir=args.dst_dir,
        num_workers=args.num_workers,
        verbose=args.verbose
    )


if __name__ == '__main__':
    main()
