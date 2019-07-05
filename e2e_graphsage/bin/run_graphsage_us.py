#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run unsupervised/semi-supervised graphsage
"""

from __future__ import division

import os
import re
import logging
import argparse
import warnings

# Data reader
import snap
import h5py
import numpy as np
import pandas as pd
from PIL import Image

# Progress logging
import tqdm
import time
import datetime

# Torch
import torch
from torch.utils.collect_env import get_pretty_env_info

# Tensorboard
from tensorboardX import SummaryWriter

# Util functions
from e2e_graphsage.utils.logging import setup_logging
from e2e_graphsage.utils.batching import batch as batch_fn

# Modelling
from e2e_graphsage.models import GAT, GraphSage, ToHierarchicalList
from e2e_graphsage.data.datasets import SnapDataset
from e2e_graphsage.data.samplers import NegativeBatchSampler
from e2e_graphsage.data.data_loaders import DataLoader2
from e2e_graphsage.losses import RollingNegativeSampler, TripletLoss

logger = logging.getLogger()

NUM_GROUP_PATTERN = '([0-9]+)'
MODEL_CHKPT_FMT = 'model.chkpt-{}.h5'.format(NUM_GROUP_PATTERN)
OPT_CHKPT_FMT = 'opt.chkpt-{}.h5'.format(NUM_GROUP_PATTERN)

LOG_STEP_COUNT_STEPS = 100
SAVE_CHECKPOINT_STEPS = 1000
VAL_STEP_COUNT_STEPS = 1000
VALID_MODES = {'train', 'val', 'pred'}


def parse_args():
    parser = argparse.ArgumentParser()

    # I/O
    parser.add_argument(
        'checkpoint_dir', metavar='CHECKPOINT_DIR', type=str,
        help='The directory to perform training in')
    parser.add_argument(
        'save_prefix', metavar='SAVE_PREFIX', type=str,
        help='Save prefix used to find graph and h5 file')

    parser.add_argument(
        '--src_node_names_file', type=str,
        help='A txt file containing starting node names')
    parser.add_argument(
        '--val_src_node_names_file', type=str,
        help='A txt file containing starting node names for validation')

    parser.add_argument(
        '--mode', type=str, default='train',
        choices=['train', 'val', 'pred'],
        help='The mode to run this script in, default train')
    parser.add_argument(
        '--pred_output_path', type=str,
        help='If mode is pred, where should embeddings be saved to?')

    # Model configs
    model_group = parser.add_argument_group('model')
    model_group.add_argument(
        '-er', '--expansion_rates', nargs='+', type=int, default=[10, 10],
        help='The expansion rate to use at each depth layer')
    model_group.add_argument(
        '-os', '--output_size', type=int, default=128,
        help='The output size of embeddings')
    model_group.add_argument(
        '-hs', '--hidden_size', type=int, default=128,
        help='The size of hidden layers in the network')
    model_group.add_argument(
        '-pm', '--pooling_method', type=str, default='max',
        choices=['max', 'min', 'mean', 'attn'],
        help='The pooling method to use')
    model_group.add_argument(
        '--num_attn_heads', type=int, default=1,
        help='The number of attention heads to use if pooling_method is attn')

    # Training configs
    train_group = parser.add_argument_group('train')
    train_group.add_argument(
        '-ti', '--train_iters', type=int, default=10000,
        help='Maximum number of iterations to train for (Only for train mode)')
    train_group.add_argument(
        '-vi', '--val_iters', type=int, default=100,
        help='Number of iterations to validate for (Only for train/val mode)')
    train_group.add_argument(
        '-b', '--batch_size', type=int, default=512,
        help='Batch size to use')
    train_group.add_argument(
        '-lr', '--learning_rate', type=float, default=3e-6,
        help='The learning rate')
    train_group.add_argument(
        '-ns', '--num_negative_samples', type=int, default=40,
        help='Number of negative samples')
    train_group.add_argument(
        '-hns', '--num_hard_negative_samples', type=int, default=5,
        help='Number of hard negative samples to select')
    train_group.add_argument(
        '-lt', '--loss_type', type=str, default='hinge',
        choices=['xent', 'cross_entropy', 'hinge', 'triplet', 'skipgram'],
        help='The type of loss function to apply')

    # Projector configs
    projector_group = parser.add_argument_group('projector')
    projector_group.add_argument(
        '--projected_embeddings_tag', type=str,
        help='Should embeddings be projected? If it should, please provide '
        'a name (Projector is only for val/pred mode)')
    projector_group.add_argument(
        '--projected_embeddings_metadata', type=str,
        help='Metadata to projected embeddings')
    projector_group.add_argument(
        '--projected_embeddings_imgs', type=str,
        help='Path to images to project images should be stored locally')

    return parser.parse_args()


def check_args(args):
    # General configs
    assert os.path.isfile(args.save_prefix + '.h5'), \
        'Unable to use save_prefix to find .h5 file'
    if args.mode == 'pred':
        assert args.pred_output_path is not None, \
            'Pred output path should be provided if mode == "pred"'
    if args.pred_output_path is not None:
        assert args.pred_output_path.endswith('.csv') or \
            args.pred_output_path.endswith('.h5'), \
            'pred_output_path should end with either .csv or .h5'

    # Model configs
    assert len(args.expansion_rates) > 0, \
        'expansion_rates has no values'

    # Train configs
    assert args.batch_size > 0, \
        'batch_size is invalid'
    assert args.num_negative_samples > 0, \
        'num_negative_samples is invalid'

    # Projector configs
    if args.projected_embeddings_tag is not None:
        assert args.src_node_names_file is not None, \
            'To project embeddings, a src_node_names_file must be provided '

    return args


def makedirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def read_node_names_from_txt_file(filename):
    node_names = []
    with open(filename, 'rb') as f:
        for line in f:
            if line.endswith(b'\n'):
                line = line[:-1]
            node_names.append(line)
    return node_names


def save_checkpoint(model, optimizer, checkpoint_dir, global_step):
    model_save_path = os.path.join(
        checkpoint_dir,
        MODEL_CHKPT_FMT.replace(NUM_GROUP_PATTERN, str(global_step))
    )
    optimizer_save_path = os.path.join(
        checkpoint_dir,
        OPT_CHKPT_FMT.replace(NUM_GROUP_PATTERN, str(global_step))
    )
    torch.save(model, model_save_path)
    torch.save(optimizer.state_dict(), optimizer_save_path)


def load_checkpoint(checkpoint_dir, global_step=None):
    # Find checkpoint files
    max_step = -1
    for f in os.listdir(checkpoint_dir):
        matches = re.match(MODEL_CHKPT_FMT, f)
        if matches is not None:
            step = int(matches.groups()[0])
            max_step = max(step, max_step)
    if max_step == -1:
        assert global_step is None, \
            'No checkpoint files are found unable to resume from checkpoint'
        return 0, None, None

    if global_step is None:
        global_step = max_step

    model_save_path = os.path.join(
        checkpoint_dir,
        MODEL_CHKPT_FMT.replace(NUM_GROUP_PATTERN, str(global_step))
    )
    optimizer_save_path = os.path.join(
        checkpoint_dir,
        OPT_CHKPT_FMT.replace(NUM_GROUP_PATTERN, str(global_step))
    )
    model = torch.load(model_save_path)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params)
    optimizer.load_state_dict(torch.load(optimizer_save_path))

    return global_step, model, optimizer


def get_or_create_model(
    checkpoint_dir,
    expansion_rates,
    input_feature_size,
    inner_feature_size=128,
    output_feature_size=128,
    pooling_method='max',
    num_attn_heads=1,
    learning_rate=3e-6
):
    global_step, model, optimizer = load_checkpoint(checkpoint_dir)
    if model is None:
        global_step = 0
        if pooling_method == 'attn':
            model = GAT(
                expansion_rates=expansion_rates,
                input_feature_size=input_feature_size,
                inner_feature_size=inner_feature_size,
                output_feature_size=output_feature_size,
                num_attn_heads=num_attn_heads,
                activation_conv=torch.nn.Tanh()
            )
        else:
            model = GraphSage(
                expansion_rates=expansion_rates,
                input_feature_size=input_feature_size,
                inner_feature_size=inner_feature_size,
                output_feature_size=output_feature_size,
                pooling_method=pooling_method
            )
        model = model.cuda()
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=learning_rate)
    else:
        for g in optimizer.param_groups:
            g['lr'] = learning_rate

    return global_step, model, optimizer


def make_data_loader(
    save_prefix,
    src_nodeids,
    G,
    num_iters,
    batch_size=512,
    num_hard_negative_samples=5
):
    assert isinstance(num_iters, int) and num_iters > 0, \
        'num_iters should be a positive integer'
    assert isinstance(batch_size, int) and batch_size > 0, \
        'batch_size should be a positive integer'

    with h5py.File(save_prefix + '.h5', 'r') as f:
        vertices_contain_labels = f['vertices_contain_labels'][()]
        node_label_ids = None
        if vertices_contain_labels:
            node_label_ids = f['node_label_ids'][:]
            node_label_ids = np.ascontiguousarray(node_label_ids[src_nodeids])

    if num_hard_negative_samples != 0:
        assert vertices_contain_labels is not None, \
            'To do hard negative sampling vertex labels should be present'

    dataset = SnapDataset(G, src_nodeids)
    batch_sampler = NegativeBatchSampler(
        dataset_len=len(dataset),
        num_iters=num_iters,
        batch_size=batch_size,
        labels=node_label_ids,
        num_hard_negative_samples=num_hard_negative_samples
    )

    return DataLoader2(
        dataset=dataset,
        batch_sampler=batch_sampler,
        pin_memory=True,
        num_workers=2
    )


def make_loss_fn(loss_type, num_negative_samples=40):
    negative_sampler = RollingNegativeSampler(
        num_negative_samples=num_negative_samples)
    loss_fn = TripletLoss(
        margin=0.2,
        loss_type=loss_type,
        distance_fn='cosine'
    )
    # if loss_type == 'hinge':
    #     loss_fn = TripletMarginLoss(margin=0.2, distance_fn='cosine')
    # else:
    #     raise NotImplementedError(
    #         'loss_type {} has not been implemented yet'.format(loss_type))
    return negative_sampler, loss_fn


def do_forward(src_nodeids, to_hierarchical_list, model):
    hierarchical_nodeids, hierarchical_input_features = \
        to_hierarchical_list(src_nodeids)
    masks = [
        (n != -1).float().unsqueeze(-1)
        for n in hierarchical_nodeids[1:]
    ]
    # masks = None
    return model(hierarchical_input_features, masks=masks)


def log_step(
    metric_dict={},
    mode='train',
    writer=None,
    global_step=0,
    elapsed_eta=None,
    training_speed=None
):
    """
    elapsed_eta is expected to be a tuple containing elapsed seconds
    and eta seconds
    """
    log_msg = '[{mode}] step: {step}'
    log_msg = log_msg.format(
        mode=mode,
        step=global_step,
    )
    for key, value in metric_dict.items():
        log_msg += ' - {}: {}'.format(key, round(value, 4))

    # Write to tensorboard
    if writer is not None:
        for key, value in metric_dict.items():
            writer.add_scalar(key, value, global_step=global_step)

    if elapsed_eta is not None:
        log_msg += ' - elapsed: {} - eta: {}'.format(
            datetime.timedelta(seconds=int(elapsed_eta[0])),
            datetime.timedelta(seconds=int(elapsed_eta[1]))
        )
        if writer is not None:
            writer.add_scalar('eta', elapsed_eta[1], global_step=global_step)

    if training_speed is not None:
        log_msg += ' - step/sec: {:.4f}'.format(training_speed)
        if writer is not None:
            writer.add_scalar(
                'step/sec', training_speed, global_step=global_step)

    logger.info(log_msg)


def do_pred(src_nodeids, to_hierarchical_list, model, batch_size=512):
    model = model.eval()

    nodeid_batches = batch_fn(src_nodeids, batch_size)
    output_embeddings = []
    desc = 'Performing inference'
    for batch in tqdm.tqdm(nodeid_batches, desc=desc, ncols=80):
        batch = torch.LongTensor(batch)
        batch_outputs = do_forward(batch, to_hierarchical_list, model)
        output_embeddings.append(batch_outputs.cpu().data.numpy())
    output_embeddings = np.concatenate(output_embeddings, axis=0)
    return output_embeddings


def do_val(
    data_loader,
    to_hierarchical_list,
    model,
    num_iters,
    num_negative_samples=40,
    loss_type='hinge',
    show_pbar=False,
    writer=None,  # For do_train
    global_step=0  # For do_train
):
    model = model.eval()

    # Make loss fn
    negative_sampler, loss_fn = \
        make_loss_fn(loss_type, num_negative_samples)

    total_loss = 0
    total_mean_rank = 0
    total_mrr = 0
    for i, batch in enumerate(data_loader, 1):
        # Transfer batch to GPU
        left_src_nodeids = batch[0].cuda(non_blocking=True)
        right_src_nodeids = batch[1].cuda(non_blocking=True)

        # Forward
        left_embeddings = \
            do_forward(left_src_nodeids, to_hierarchical_list, model)
        right_embeddings = \
            do_forward(right_src_nodeids, to_hierarchical_list, model)
        loss, mean_rank, mrr = loss_fn(
            left_embeddings,
            right_embeddings,
            negative_sampler(left_embeddings),
            loss_only=False
        )

        total_loss += loss.item()
        total_mean_rank += mean_rank.item()
        total_mrr += mrr.item()

    metric_dict = {
        loss_type: total_loss / i,
        'mean_rank': total_mean_rank / i,
        'mrr': total_mrr / i
    }

    # Log results
    log_step(
        metric_dict=metric_dict,
        mode='val',
        writer=writer,
        global_step=global_step
    )


def do_train(
    checkpoint_dir,
    data_loader,
    val_data_loader,
    to_hierarchical_list,
    model,
    optimizer,
    start_step=0,
    train_iters=10000,
    val_iters=1000,
    num_negative_samples=40,
    loss_type='hinge'
):
    model = model.train()

    # Make loss fn
    negative_sampler, loss_fn = \
        make_loss_fn(loss_type, num_negative_samples)

    # Make tensorboard writer
    writer = SummaryWriter(checkpoint_dir)
    if val_data_loader is not None:
        val_dir = checkpoint_dir + '_val'
        val_writer = SummaryWriter(val_dir)

    start_time = time.time()
    t0 = time.time()
    for i, batch in enumerate(data_loader, start_step):
        # Transfer batch to GPU
        left_src_nodeids = batch[0].cuda(non_blocking=True)
        right_src_nodeids = batch[1].cuda(non_blocking=True)

        # Determine if should do val
        is_last_step = i == train_iters - 1
        should_save_checkpoint = i % SAVE_CHECKPOINT_STEPS == 0
        should_do_logging = i % LOG_STEP_COUNT_STEPS == 0
        should_do_val = (i % VAL_STEP_COUNT_STEPS == 0 or is_last_step) \
            and val_data_loader is not None

        # Forward
        left_embeddings = \
            do_forward(left_src_nodeids, to_hierarchical_list, model)
        right_embeddings = \
            do_forward(right_src_nodeids, to_hierarchical_list, model)

        if should_do_val or should_do_logging:
            loss, mean_rank, mrr = loss_fn(
                left_embeddings,
                right_embeddings,
                negative_sampler(left_embeddings),
                loss_only=False
            )
        else:
            loss = loss_fn(
                left_embeddings,
                right_embeddings,
                negative_sampler(left_embeddings),
                loss_only=True
            )

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if should_do_val:
            # Do validation
            do_val(
                data_loader=val_data_loader,
                to_hierarchical_list=to_hierarchical_list,
                model=model,
                num_iters=val_iters,
                num_negative_samples=num_negative_samples,
                loss_type=loss_type,
                show_pbar=False,
                writer=val_writer,
                global_step=i
            )
            model = model.train()

        if should_save_checkpoint:
            # Save checkpoint
            save_checkpoint(model, optimizer, checkpoint_dir, i)

        if should_do_logging:
            metric_dict = {
                loss_type: loss.item(),
                'mean_rank': mean_rank.item(),
                'mrr': mrr.item()
            }

            # Compute training speed and reset timer
            if i - start_step > 1000:
                training_speed = LOG_STEP_COUNT_STEPS / (time.time() - t0)
                elapsed_seconds = time.time() - start_time
                eta_seconds = elapsed_seconds / (i + 1 - start_step) * \
                    (train_iters - i - 1)
                elapsed_eta = (elapsed_seconds, eta_seconds)
            else:
                # don't log speed for starting steps
                training_speed = None
                elapsed_eta = None
            t0 = time.time()

            log_step(
                metric_dict=metric_dict,
                mode='train',
                writer=writer,
                global_step=i,
                elapsed_eta=elapsed_eta,
                training_speed=training_speed
            )


def project_embeddings(
    checkpoint_dir,
    projected_embeddings_tag,
    projected_embeddings_metadata,
    projected_embeddings_imgs,
    embeddings
):
    if len(embeddings) >= 100000:
        logger.warn('Large number of embeddings about to be projected')

    metadata = None
    metadata_header = None
    if projected_embeddings_metadata is not None:
        metadata_df = pd.read_csv(projected_embeddings_metadata)
        metadata_header = metadata_df.columns.tolist()
        metadata = metadata_df.values.tolist()

    imgs = None
    if projected_embeddings_imgs is not None:
        image_filenames = pd.read_csv(
            projected_embeddings_imgs, header=-1)[0]
        imgs = []
        desc = 'Reading images to form sprites'
        for img_file in tqdm.tqdm(image_filenames, desc=desc, ncols=80):
            imgs.append(np.array(
                Image.open(img_file).resize((32, 32))
            ) / 255.0)
        imgs = torch.Tensor(imgs).permute(0, 3, 1, 2)

    writer = SummaryWriter(checkpoint_dir)
    writer.add_embedding(
        embeddings,
        metadata=metadata,
        label_img=imgs,
        tag=projected_embeddings_tag,
        metadata_header=metadata_header
    )
    writer.close()


def save_embeddings(pred_output_path, node_names, embeddings):
    if pred_output_path.endswith('.h5'):
        with h5py.File(pred_output_path, 'w') as f:
            f.create_dataset(
                'node_name',
                shape=(len(node_names),),
                dtype=h5py.special_dtype(vlen=bytes),
                data=node_names
            )
            f['embeddings'] = embeddings
    else:
        if not pred_output_path.endswith('.csv'):
            warning_msg = (
                'pred_output_path not recognized must be a csv or h5 file. '
                'Still saving as csv file'
            )
            warnings.warn(warning_msg)

        embeddings_df = pd.DataFrame(
            node_names, columns=['node_name'])
        for i in range(embeddings.shape[1]):
            embeddings_df[i] = embeddings[:, i]
        embeddings_df.to_csv(pred_output_path, index=False, header=False)


def run_graphsage_us(
    checkpoint_dir,
    save_prefix,
    src_node_names,
    val_node_names=None,
    mode='train',
    pred_output_path=None,
    expansion_rates=[10, 10],
    output_size=128,
    hidden_size=128,
    pooling_method='max',
    num_attn_heads=1,
    train_iters=10000,
    val_iters=1000,
    batch_size=512,
    learning_rate=3e-6,
    num_negative_samples=40,
    num_hard_negative_samples=5,
    loss_type='hinge',
    projected_embeddings_tag=None,
    projected_embeddings_metadata=None,
    projected_embeddings_imgs=None
):
    """
    Main function for running training validation and prediction task for
    unsupervised graphsage
    """
    assert mode in VALID_MODES, \
        'mode is invalid should be one of {}'.format(VALID_MODES)
    if mode != 'train':
        if val_node_names is not None:
            warnings.warn('val_node_names is not used in {} mode'.format(mode))
            if mode == 'val':
                warnings.warn('For val mode, instead use src_node_names')
        val_node_names = None

    # Read graph from save prefix
    logger.info('Getting adjacency_list and input_features')
    with h5py.File(save_prefix + '.h5', 'r') as f:
        node_names = f['node_names'][:]
        adjacency_list = f['adjacency_list'][:]
        node_features = f['node_features'][:]
        unidirectional_edges = f['unidirectional_edges'][()]

    if unidirectional_edges:
        G = snap.TNGraph.Load(snap.TFIn(save_prefix + '.graph'))
    else:
        G = snap.TUNGraph.Load(snap.TFIn(save_prefix + '.graph'))
    G.Defrag()
    G.GetNodes(), G.GetEdges()

    # Identify src_nodeids and val_nodeids
    logger.info('Parsing nodeids')
    node_name_to_idx = dict(zip(node_names, range(len(node_names))))
    if src_node_names is None:
        src_nodeids = list(range(len(node_names)))
    else:
        src_nodeids = [node_name_to_idx[name] for name in src_node_names]
    if val_node_names is not None:
        val_nodeids = [node_name_to_idx[name] for name in val_node_names]

    # model and optimizer
    logger.info('Creating model')
    to_hierarchical_list = ToHierarchicalList(
        adjacency_list=adjacency_list,
        input_features=node_features,
        expansion_rates=expansion_rates
    ).cuda()
    start_step, model, optimizer = get_or_create_model(
        checkpoint_dir=checkpoint_dir,
        expansion_rates=expansion_rates,
        input_feature_size=node_features.shape[1],
        inner_feature_size=hidden_size,
        output_feature_size=output_size,
        pooling_method=pooling_method,
        num_attn_heads=num_attn_heads,
        learning_rate=learning_rate
    )

    logger.info('Mode - {}'.format(mode))
    if mode == 'train':
        if train_iters - start_step == 0:
            logger.warn('Training already reached {} step'.format(train_iters))
            logger.warn('No further training performed')
            return

        logger.info('Creating data loader')
        data_loader = make_data_loader(
            save_prefix=save_prefix,
            src_nodeids=src_nodeids,
            G=G,
            num_iters=train_iters - start_step,
            batch_size=batch_size,
            num_hard_negative_samples=num_hard_negative_samples
        )
        val_data_loader = None
        if val_node_names is not None:
            val_data_loader = make_data_loader(
                save_prefix=save_prefix,
                src_nodeids=val_nodeids,
                G=G,
                num_iters=val_iters,
                batch_size=batch_size,
                num_hard_negative_samples=num_hard_negative_samples
            )

        logger.info('Ready for do train')
        do_train(
            checkpoint_dir=checkpoint_dir,
            data_loader=data_loader,
            val_data_loader=val_data_loader,
            to_hierarchical_list=to_hierarchical_list,
            model=model,
            optimizer=optimizer,
            start_step=start_step,
            train_iters=train_iters,
            val_iters=val_iters,
            num_negative_samples=num_negative_samples,
            loss_type=loss_type
        )

    elif mode == 'val':
        logger.info('Creating data loader')
        data_loader = make_data_loader(
            save_prefix=save_prefix,
            src_nodeids=src_nodeids,
            G=G,
            num_iters=val_iters,
            batch_size=batch_size,
            num_hard_negative_samples=num_hard_negative_samples
        )

        logger.info('Performing validation')
        do_val(
            data_loader=data_loader,
            to_hierarchical_list=to_hierarchical_list,
            model=model,
            num_iters=val_iters,
            num_negative_samples=num_negative_samples,
            loss_type=loss_type,
            show_pbar=True
        )

        if projected_embeddings_tag is None:
            return None

        logger.info('Generating embeddings')
        output_embeddings = do_pred(
            src_nodeids=src_nodeids,
            to_hierarchical_list=to_hierarchical_list,
            model=model,
            batch_size=batch_size
        )

        logger.info('Projecting embeddings')
        project_embeddings(
            checkpoint_dir=checkpoint_dir,
            projected_embeddings_tag=projected_embeddings_tag,
            projected_embeddings_metadata=projected_embeddings_metadata,
            projected_embeddings_imgs=projected_embeddings_imgs,
            embeddings=output_embeddings
        )

    elif mode == 'pred':
        assert pred_output_path is not None, \
            'pred_output_path has to be set for pred mode'

        logger.info('Generating embeddings')
        output_embeddings = do_pred(
            src_nodeids=src_nodeids,
            to_hierarchical_list=to_hierarchical_list,
            model=model,
            batch_size=batch_size
        )

        if projected_embeddings_tag is not None:
            logger.info('Projecting embeddings')
            project_embeddings(
                checkpoint_dir=checkpoint_dir,
                projected_embeddings_tag=projected_embeddings_tag,
                projected_embeddings_metadata=projected_embeddings_metadata,
                projected_embeddings_imgs=projected_embeddings_imgs,
                embeddings=output_embeddings
            )

        logger.info('Saving embeddings to output file')
        save_embeddings(
            pred_output_path=pred_output_path,
            node_names=src_node_names,
            embeddings=output_embeddings
        )

    else:
        err_msg = 'mode is not recognized, {}'.format(mode)
        raise ValueError(err_msg)


def main():
    args = check_args(parse_args())
    makedirs(args.checkpoint_dir)
    setup_logging(os.path.join(
        args.checkpoint_dir,
        '{}.log'.format(args.mode)
    ), 'a')

    logger.info('Getting environmental info')
    logger.info(get_pretty_env_info() + '\n')

    logger.info('Getting script arguments')
    logger.info(args.__repr__() + '\n')

    logger.info('Reading node names')
    src_node_names = None
    val_node_names = None
    if args.src_node_names_file is not None:
        src_node_names = read_node_names_from_txt_file(
            args.src_node_names_file)
    if args.val_src_node_names_file is not None:
        val_node_names = read_node_names_from_txt_file(
            args.val_src_node_names_file)

    run_graphsage_us(
        checkpoint_dir=args.checkpoint_dir,
        save_prefix=args.save_prefix,
        src_node_names=src_node_names,
        val_node_names=val_node_names,
        mode=args.mode,
        pred_output_path=args.pred_output_path,
        expansion_rates=args.expansion_rates,
        output_size=args.output_size,
        hidden_size=args.hidden_size,
        pooling_method=args.pooling_method,
        num_attn_heads=args.num_attn_heads,
        train_iters=args.train_iters,
        val_iters=args.val_iters,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_negative_samples=args.num_negative_samples,
        num_hard_negative_samples=args.num_hard_negative_samples,
        loss_type=args.loss_type,
        projected_embeddings_tag=args.projected_embeddings_tag,
        projected_embeddings_metadata=args.projected_embeddings_metadata,
        projected_embeddings_imgs=args.projected_embeddings_imgs
    )


if __name__ == "__main__":
    main()
