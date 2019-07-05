#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run graphsage for single class classification
"""

from __future__ import division

import os
import re
import logging
import argparse
import warnings

# Data reader
import h5py
import numpy as np
import pandas as pd
# from PIL import Image

# Progress logging
import tqdm
import time
import datetime

# Torch
import torch
from torch.utils.collect_env import get_pretty_env_info

# Tensorboard
from tensorboardX import SummaryWriter

# Sklearn
from sklearn import metrics as skmetrics

# Matplotlib
from matplotlib import pyplot as plt

# Util functions
from e2e_graphsage.utils.logging import setup_logging
from e2e_graphsage.utils.batching import batch as batch_fn

# Modelling
from e2e_graphsage.models import GAT, GraphSage, ToHierarchicalList
from e2e_graphsage.data.data_loaders import DataLoader2

plt.switch_backend('agg')
logger = logging.getLogger()

NUM_GROUP_PATTERN = '([0-9]+)'
MODEL_CHKPT_FMT = 'model.chkpt-{}.h5'.format(NUM_GROUP_PATTERN)
OPT_CHKPT_FMT = 'opt.chkpt-{}.h5'.format(NUM_GROUP_PATTERN)

LOG_STEP_COUNT_STEPS = 500
SAVE_CHECKPOINT_STEPS = 5000
VAL_STEP_COUNT_STEPS = 5000
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
        '-b', '--batch_size', type=int, default=512,
        help='Batch size to use')
    train_group.add_argument(
        '-lr', '--learning_rate', type=float, default=3e-6,
        help='The learning rate')
    train_group.add_argument(
        '-lt', '--loss_type', type=str, default='xent',
        choices=['xent', 'cross_entropy', 'bce', 'mse', 'l2'],
        help='The type of loss function to apply')

    # Validation configs
    val_group = parser.add_argument_group('val')
    val_group.add_argument(
        '-cmt', '--confusion_matrix_tag', type=str,
        help='Should confusion matrix be drawn? If it should, please provide '
        'a name')

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
    num_classes,
    input_feature_size,
    inner_feature_size=128,
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
                output_feature_size=num_classes,
                num_attn_heads=num_attn_heads,
                activation_conv=torch.nn.Tanh()
            )
        else:
            model = GraphSage(
                expansion_rates=expansion_rates,
                input_feature_size=input_feature_size,
                inner_feature_size=inner_feature_size,
                output_feature_size=num_classes,
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
    src_nodeids,
    node_label_ids,
    num_iters=10000,
    batch_size=512
):
    """
    Makes a data loader that generates batches randomly
    """
    dataset = torch.utils.data.TensorDataset(
        torch.LongTensor(src_nodeids),
        torch.LongTensor(node_label_ids)
    )
    sampler = torch.utils.data.RandomSampler(
        dataset,
        replacement=True,
        num_samples=num_iters * batch_size
    )
    return DataLoader2(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=True
    )


def one_hot_encode(labels, num_classes):
    labels = labels.reshape(-1, 1)
    one_hot_encodings = torch.zeros(
        (labels.shape[0], num_classes),
        device=labels.device,
        dtype=torch.float32
    ).scatter(1, labels, 1)
    return one_hot_encodings


def compute_loss(logits, labels, num_classes, loss_type='xent'):
    if loss_type in {'xent', 'cross_entropy', 'bce'}:
        loss = torch.nn.functional.cross_entropy(logits, labels)
    elif loss_type in {'mse', 'l2'}:
        loss = torch.nn.functional.mse_loss(
            torch.nn.functional.sigmoid(logits),
            one_hot_encode(labels, num_classes)
        )
    else:
        raise ValueError('loss_type {} not yet implemented'.format(loss_type))
    return loss


def do_forward(src_nodeids, to_hierarchical_list, model):
    hierarchical_nodeids, hierarchical_input_features = \
        to_hierarchical_list(src_nodeids)
    masks = [
        (n != -1).float().unsqueeze(-1)
        for n in hierarchical_nodeids[1:]
    ]
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


def draw_confusion_matrix(preds, labels, label_names=None, normalize=False):
    assert isinstance(preds, np.ndarray) and isinstance(labels, np.ndarray)

    C = skmetrics.confusion_matrix(labels, preds)
    if normalize:
        C = C.astype(np.float32) / C.sum(axis=1)[:, None]

    num_labels = len(C)
    thresh = (C.max() - C.min()) / 2.0

    fig = plt.figure(dpi=20 * num_labels)

    plt.imshow(C, cmap=plt.cm.Blues)

    plt.xticks(range(num_labels), label_names, rotation=90, size=6)
    plt.yticks(range(num_labels), label_names, rotation=0, size=6)

    plt.xlabel('Pred label')
    plt.ylabel('True label')

    for i in range(num_labels):
        for j in range(num_labels):
            value = '{:.4f}'.format(C[i, j])
            color = 'black'
            if C[i, j] > thresh:
                color = 'white'
            plt.text(
                j, i, value,
                ha="center", va="center",
                color=color, size=40 / num_labels
            )

    plt.tight_layout()
    return fig


def do_pred(
    src_nodeids,
    to_hierarchical_list,
    model,
    batch_size=512,
    show_pbar=False
):
    model = model.eval()

    nodeid_batches = batch_fn(src_nodeids, batch_size)
    output_preds = []
    desc = 'Performing inference'
    if show_pbar:
        nodeid_batches = tqdm.tqdm(nodeid_batches, desc=desc, ncols=80)
    for batch in nodeid_batches:
        batch = torch.LongTensor(batch)
        batch_outputs = do_forward(batch, to_hierarchical_list, model)
        output_preds.append(batch_outputs.cpu().data)
    output_preds = torch.cat(output_preds, dim=0)
    return output_preds


def do_val(
    src_nodeids,
    node_label_ids,
    to_hierarchical_list,
    model,
    num_classes,
    batch_size=512,
    loss_type='xent',
    label_names=None,
    confusion_matrix_tag=None,
    show_pbar=False,
    writer=None,  # For do_train
    global_step=0  # For do_train
):
    model = model.eval()

    # Forward
    output_preds = do_pred(
        src_nodeids=src_nodeids,
        to_hierarchical_list=to_hierarchical_list,
        model=model,
        batch_size=batch_size
    )

    # Compute loss
    loss = compute_loss(
        logits=output_preds,
        labels=node_label_ids,
        num_classes=num_classes,
        loss_type='xent'
    )

    # Compute val metrics
    output_labels = output_preds.argmax(dim=1)
    acc = (output_labels == node_label_ids).float().mean()

    indices_of_ranks = output_preds.argsort(dim=1, descending=True)
    ranks = torch.matmul(
        (indices_of_ranks == node_label_ids[:, None]).float(),
        torch.arange(
            num_classes,
            dtype=torch.float32,
            device=indices_of_ranks.device
        )
    ) + 1
    mrr = (1 / ranks).mean()
    mean_rank = ranks.mean()

    metric_dict = {
        loss_type: loss.item(),
        'acc': acc.item(),
        'mrr': mrr.item(),
        'mean_rank': mean_rank.item()
    }

    # Log results
    log_step(
        metric_dict=metric_dict,
        mode='val',
        writer=writer,
        global_step=global_step
    )

    if confusion_matrix_tag is not None:
        assert writer is not None
        fig = draw_confusion_matrix(
            preds=output_labels.cpu().data.numpy(),
            labels=node_label_ids.cpu().data.numpy(),
            label_names=label_names,
            normalize=True
        )
        writer.add_figure(
            confusion_matrix_tag,
            fig,
            global_step=global_step
        )


def do_train(
    checkpoint_dir,
    data_loader,
    to_hierarchical_list,
    model,
    optimizer,
    num_classes,
    start_step=0,
    train_iters=10000,
    loss_type='xent',
    val_src_nodeids=None,
    val_node_label_ids=None,
    label_names=None,
    confusion_matrix_tag=None
):
    model = model.train()

    # Make tensorboard writer
    writer = SummaryWriter(checkpoint_dir)
    if val_src_nodeids is not None:
        val_dir = checkpoint_dir + '_val'
        val_writer = SummaryWriter(val_dir)

    start_time = time.time()
    t0 = time.time()
    for i, batch in enumerate(data_loader, start_step):
        src_nodeids = batch[0].cuda(non_blocking=True)
        node_label_ids = batch[1].cuda(non_blocking=True)

        # Determine if should do val
        is_last_step = i == train_iters - 1
        should_save_checkpoint = i % SAVE_CHECKPOINT_STEPS == 0
        should_do_logging = i % LOG_STEP_COUNT_STEPS == 0
        should_do_val = (i % VAL_STEP_COUNT_STEPS == 0 or is_last_step) \
            and val_src_nodeids is not None

        # Forward
        output_preds = do_forward(src_nodeids, to_hierarchical_list, model)
        loss = compute_loss(
            logits=output_preds,
            labels=node_label_ids,
            num_classes=num_classes,
            loss_type=loss_type
        )

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if should_do_val:
            # Do validation
            do_val(
                src_nodeids=val_src_nodeids,
                node_label_ids=val_node_label_ids,
                to_hierarchical_list=to_hierarchical_list,
                model=model,
                num_classes=num_classes,
                batch_size=len(src_nodeids),
                loss_type=loss_type,
                label_names=label_names,
                confusion_matrix_tag=confusion_matrix_tag,
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
                loss_type: loss.item()
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


def save_outputs(pred_output_path, node_names, output_preds):
    output_preds = output_preds.numpy()
    if pred_output_path.endswith('.h5'):
        with h5py.File(pred_output_path, 'w') as f:
            f.create_dataset(
                'node_name',
                shape=(len(node_names),),
                dtype=h5py.special_dtype(vlen=bytes),
                data=node_names
            )
            f['logits'] = output_preds
    else:
        if not pred_output_path.endswith('.csv'):
            warning_msg = (
                'pred_output_path not recognized must be a csv or h5 file. '
                'Still saving as csv file'
            )
            warnings.warn(warning_msg)

        preds_df = pd.DataFrame()
        preds_df['node_name'] = node_names
        for i in range(output_preds.shape[1]):
            preds_df[i] = output_preds[:, i]
        preds_df.to_csv(pred_output_path, index=False, header=False)


def run_graphsage_cls(
    checkpoint_dir,
    save_prefix,
    src_node_names=None,
    val_node_names=None,
    mode='train',
    pred_output_path=None,
    expansion_rates=[10, 10],
    hidden_size=128,
    pooling_method='max',
    num_attn_heads=1,
    train_iters=10000,
    batch_size=512,
    learning_rate=3e-6,
    loss_type='xent',
    confusion_matrix_tag=None
):
    """
    Main function for running training validation and prediction task for
    single class classification with graphsage
    """
    assert mode in VALID_MODES, \
        'mode is invalid should be one of {}'.format(VALID_MODES)
    if mode != 'train':
        if val_node_names is not None:
            warnings.warn('val_node_names is not used in {} mode'.format(mode))
            if mode == 'val':
                warnings.warn('For val mode, instead use src_node_names')
        val_node_names = None

    # Read graph data from save prefix
    logger.info('Getting adjacency_list and input_features')
    with h5py.File(save_prefix + '.h5', 'r') as f:
        node_names = f['node_names'][:]
        adjacency_list = f['adjacency_list'][:]
        node_features = f['node_features'][:]

    # Read labels if mode is not pred
    num_classes = 1
    if mode != 'pred':
        with h5py.File(save_prefix + '.h5', 'r') as f:
            assert f['vertices_contain_labels'][()], \
                '{} does not contain labels'.format(save_prefix)
            num_classes = f['num_classes'][()]
            label_names = f['label_names'][:]
            all_node_label_ids = f['node_label_ids'][:]

    # Identify src_nodeids and val_nodeids
    logger.info('Parsing nodeids')
    node_name_to_idx = dict(zip(node_names, range(len(node_names))))
    if src_node_names is None:
        src_node_names = node_names
        src_nodeids = list(range(len(node_names)))
    else:
        src_nodeids = [node_name_to_idx[name] for name in src_node_names]
    src_nodeids = torch.LongTensor(src_nodeids)
    if val_node_names is not None:
        val_nodeids = [node_name_to_idx[name] for name in val_node_names]
        val_nodeids = torch.LongTensor(val_nodeids)

    if mode != 'pred':
        src_node_label_ids = all_node_label_ids[src_nodeids]
        src_node_label_ids = torch.LongTensor(src_node_label_ids)
        if val_node_names is not None:
            val_node_label_ids = all_node_label_ids[val_nodeids]
            val_node_label_ids = torch.LongTensor(val_node_label_ids)

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
        num_classes=num_classes,
        input_feature_size=node_features.shape[1],
        inner_feature_size=hidden_size,
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
            src_nodeids=src_nodeids,
            node_label_ids=src_node_label_ids,
            num_iters=train_iters - start_step,
            batch_size=batch_size
        )

        logger.info('Ready for do train')
        do_train(
            checkpoint_dir=checkpoint_dir,
            data_loader=data_loader,
            to_hierarchical_list=to_hierarchical_list,
            model=model,
            optimizer=optimizer,
            num_classes=num_classes,
            start_step=start_step,
            train_iters=train_iters,
            loss_type=loss_type,
            val_src_nodeids=val_nodeids,
            val_node_label_ids=val_node_label_ids,
            label_names=label_names,
            confusion_matrix_tag=confusion_matrix_tag
        )

    elif mode == 'val':
        assert start_step != 0, \
            'No checkpoint found and thus loaded'

        logger.info('Performing validation')
        do_val(
            src_nodeids=src_nodeids,
            node_label_ids=src_node_label_ids,
            to_hierarchical_list=to_hierarchical_list,
            model=model,
            num_classes=num_classes,
            batch_size=batch_size,
            loss_type=loss_type,
            label_names=label_names,
            confusion_matrix_tag=confusion_matrix_tag,
            show_pbar=True
        )

    elif mode == 'pred':
        assert pred_output_path is not None, \
            'pred_output_path has to be set for pred_mode'
        assert start_step != 0, \
            'No checkpoint found and thus loaded'

        logger.info('Making predictions')
        output_preds = do_pred(
            src_nodeids=src_nodeids,
            to_hierarchical_list=to_hierarchical_list,
            mdoel=model,
            batch_size=batch_size
        )

        logger.info('Saving predictions to output file')
        save_outputs(
            pred_output_path=pred_output_path,
            node_names=src_node_names,
            output_preds=output_preds
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

    run_graphsage_cls(
        checkpoint_dir=args.checkpoint_dir,
        save_prefix=args.save_prefix,
        src_node_names=src_node_names,
        val_node_names=val_node_names,
        mode=args.mode,
        pred_output_path=args.pred_output_path,
        expansion_rates=args.expansion_rates,
        hidden_size=args.hidden_size,
        pooling_method=args.pooling_method,
        num_attn_heads=args.num_attn_heads,
        train_iters=args.train_iters,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        loss_type=args.loss_type,
        confusion_matrix_tag=args.confusion_matrix_tag
    )


if __name__ == "__main__":
    main()
