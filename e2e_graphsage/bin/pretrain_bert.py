#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pretraining of BERT
Taken from https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/lm_finetuning/simple_lm_finetuning.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import io
import re
import os
import sys
# import time
import random
import logging
import argparse

from tqdm import tqdm

import numpy as np

import torch
import torch.utils.data
from torch.utils.collect_env import get_pretty_env_info

from tensorboardX import SummaryWriter

from pytorch_pretrained_bert import WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertConfig, BertForPreTraining
from pytorch_pretrained_bert.optimization import BertAdam

from e2e_graphsage.utils.tokenization import BertTokenizer
from e2e_graphsage.utils.logging import setup_logging

logger = logging.getLogger(__name__)

# The way that special tokens are handled are pretty ugly
# maybe imporove this in the future?
SEP_TOKEN = '<sep>'
CLS_TOKEN = '<cls>'
MASK_TOKEN = '<mask>'


class BertDataset(torch.utils.data.Dataset):

    ___files_initialized = False

    def __init__(
        self,
        corpus_path,
        tokenizer,
        seq_len,
        encoding='utf-8',
        corpus_lines=None,
        on_memory=False,
        pad_to_seq_len=True
    ):
        self.tokenizer = tokenizer
        self.tokenizer_num_tokens = len(tokenizer.vocab)

        for token in SEP_TOKEN, CLS_TOKEN, MASK_TOKEN:
            assert token in tokenizer.vocab, \
                "'{}' is a required token in sentencepiece model".format(token)
        self.sep_token_id = self.tokenizer.vocab[SEP_TOKEN]
        self.cls_token_id = self.tokenizer.vocab[CLS_TOKEN]
        self.mask_token_id = self.tokenizer.vocab[MASK_TOKEN]

        self.pad_to_seq_len = pad_to_seq_len
        self.seq_len = seq_len
        self.on_memory = on_memory
        self.corpus_lines = corpus_lines  # number of non-empty lines in
        # input corpus
        self.corpus_path = corpus_path
        self.encoding = encoding
        self.current_doc = 0  # to avoid random sentence from same doc

        # for loading samples directly from file
        self.sample_counter = 0  # used to keep track of full epochs on file
        self.line_buffer = None  # keep second sentence of a pair in memory and
        # use as first sentence in next pair

        # for loading samples in memory
        self.current_random_doc = 0
        self.num_docs = 0
        self.sample_to_doc = []  # map sample index to doc and line

        # load samples into memory
        if on_memory:
            self.all_docs = []
            doc = []
            self.corpus_lines = 0
            with io.open(corpus_path, "r", encoding=encoding) as f:
                desc = "Loading Dataset"
                for line in tqdm(f, desc=desc, total=corpus_lines):
                    line = line.strip()
                    if line == "":
                        self.all_docs.append(doc)
                        doc = []
                        # remove last added sample because there won't be a
                        # subsequent line anymore in the doc
                        self.sample_to_doc.pop()
                    else:
                        # store as one sample
                        sample = {"doc_id": len(self.all_docs),
                                  "line": len(doc)}
                        self.sample_to_doc.append(sample)
                        doc.append(line)
                        self.corpus_lines = self.corpus_lines + 1

            # if last row in file is not empty
            if self.all_docs[-1] != doc:
                self.all_docs.append(doc)
                self.sample_to_doc.pop()

            self.num_docs = len(self.all_docs)

        # load samples later lazily from disk
        else:
            if self.corpus_lines is None:
                with io.open(corpus_path, "r", encoding=encoding) as f:
                    self.corpus_lines = 0
                    desc = "Loading Dataset"
                    for line in tqdm(f, desc=desc, total=corpus_lines):
                        if line.strip() == "":
                            self.num_docs += 1
                        else:
                            self.corpus_lines += 1

                    # if doc does not end with empty line
                    if line.strip() != "":
                        self.num_docs += 1

            # self.file = io.open(corpus_path, "r", encoding=encoding)
            # self.random_file = io.open(corpus_path, "r", encoding=encoding)

    def __len__(self):
        # last line of doc won't be used, because there's no "nextSentence".
        # Additionally, we start counting at 0.
        return self.corpus_lines - self.num_docs - 1

    def __getitem__(self, index):
        # open files here to allow for multiprocessing
        if not self.___files_initialized:
            self.___files_initialized = True
            if not self.on_memory:
                self.file = \
                    io.open(self.corpus_path, "r", encoding=self.encoding)
                self.random_file = \
                    io.open(self.corpus_path, "r", encoding=self.encoding)

                start_line = \
                    random.randint(0, min(self.corpus_lines - 1, 10000))
                desc = 'Opening file to random line'
                for i in tqdm(range(start_line), desc=desc):
                    self.__getitem__(i)

        cur_id = self.sample_counter
        self.sample_counter += 1
        if not self.on_memory:
            # after one epoch we start again from beginning of file
            if cur_id != 0 and (cur_id % len(self) == 0):
                self.file.close()
                self.file = \
                    io.open(self.corpus_path, "r", encoding=self.encoding)

        t1, t2, is_next_label = self.random_sent(index)

        input_ids, input_mask, segment_ids, lm_label_ids = \
            self.convert_sample_to_features(t1, t2)

        return (
            torch.tensor(input_ids),
            torch.tensor(input_mask),
            torch.tensor(segment_ids),
            torch.tensor(lm_label_ids),
            torch.tensor(is_next_label)
        )

    def random_sent(self, index):
        """
        Get one sample from corpus consisting of two sentences. With prob. 50%
        these are two subsequent sentences from one doc. With 50% the second
        sentence will be a random one from another doc.
        :param index: int, index of sample.
        :return: (str, str, int), sentence 1, sentence 2, isNextSentence Label
        """
        t1, t2 = self.get_corpus_line(index)
        if random.random() > 0.5:
            label = 0
        else:
            t2 = self.get_random_line()
            label = 1

        assert len(t1) > 0
        assert len(t2) > 0
        return t1, t2, label

    def get_corpus_line(self, index):
        """
        Get one sample from corpus consisting of a pair of two subsequent lines
        from the same doc.
        :param index: int, index of sample.
        :return: (str, str), two subsequent sentences from corpus
        """
        t1 = ""
        t2 = ""
        assert index < self.corpus_lines
        if self.on_memory:
            sample = self.sample_to_doc[index]
            t1 = self.all_docs[sample["doc_id"]][sample["line"]]
            t2 = self.all_docs[sample["doc_id"]][sample["line"] + 1]
            # used later to avoid random nextSentence from same doc
            self.current_doc = sample["doc_id"]
            return t1, t2
        else:
            if self.line_buffer is None:
                # read first non-empty line of file
                while t1 == "":
                    t1 = next(self.file).strip()
                    t2 = next(self.file).strip()
            else:
                # use t2 from previous iteration as new t1
                t1 = self.line_buffer
                t2 = next(self.file).strip()
                # skip empty rows that are used for separating documents and
                # keep track of current doc id
                while t2 == "" or t1 == "":
                    t1 = next(self.file).strip()
                    t2 = next(self.file).strip()
                    self.current_doc = self.current_doc + 1
            self.line_buffer = t2

        assert t1 != ""
        assert t2 != ""
        return t1, t2

    def get_random_line(self):
        """
        Get random line from another document for nextSentence task.
        :return: str, content of one line
        """
        # Similar to original tf repo: This outer loop should rarely go for
        # more than one iteration for large corpora. However, just to be
        # careful, we try to make sure that the random document is not the
        # same as the document we're processing.
        for _ in range(10):
            if self.on_memory:
                rand_doc_idx = random.randint(0, len(self.all_docs) - 1)
                rand_doc = self.all_docs[rand_doc_idx]
                line = rand_doc[random.randrange(len(rand_doc))]
            else:
                rand_index = random.randint(
                    1, self.corpus_lines if self.corpus_lines < 1000 else 1000)
                # pick random line
                for _ in range(rand_index):
                    line = self.get_next_line()
            # check if our picked random line is really from another doc like
            # we want it to be
            if self.current_random_doc != self.current_doc:
                break
        return line

    def get_next_line(self):
        """
        Gets next line of random_file and starts over when reaching end of file
        """
        try:
            line = next(self.random_file).strip()
            # keep track of which document we are currently looking at to
            # later avoid having the same doc as t1
            if line == "":
                self.current_random_doc = self.current_random_doc + 1
                line = next(self.random_file).strip()
        except StopIteration:
            self.random_file.close()
            self.random_file = \
                io.open(self.corpus_path, "r", encoding=self.encoding)
            line = next(self.random_file).strip()
        return line

    def convert_sample_to_features(self, t1, t2):
        """
        Convert a raw sample (pair of sentences as tokenized strings) into a
        proper training sample with IDs, LM labels, input_mask, CLS, SEP,
        tokens etc.
        """

        token_ids_a = self.tokenizer.tokenize_as_ids(t1)
        token_ids_b = self.tokenizer.tokenize_as_ids(t2)
        assert len(token_ids_b) > 0

        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for CLS, SEP, SEP with "- 3"
        self._truncate_seq_pair(token_ids_a, token_ids_b)

        token_ids_a, t1_label = self._do_lm_preproc(token_ids_a)
        token_ids_b, t2_label = self._do_lm_preproc(token_ids_b)

        # concatenate sequence pairs with CLS, SEP, SEP
        input_ids = [self.cls_token_id] + token_ids_a + [self.sep_token_id] + \
            token_ids_b + [self.sep_token_id]
        lm_label_ids = [-1] + t1_label + [-1] + t2_label + [-1]
        segment_ids = [0] * (len(token_ids_a) + 2) + \
            [1] * (len(token_ids_b) + 1)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        if self.pad_to_seq_len:
            input_ids += [0] * (self.seq_len - len(input_ids))
            input_mask += [0] * (self.seq_len - len(input_mask))
            segment_ids += [0] * (self.seq_len - len(segment_ids))
            lm_label_ids += [-1] * (self.seq_len - len(lm_label_ids))

        input_ids_len = len(input_ids)
        assert len(input_mask) == input_ids_len
        assert len(segment_ids) == input_ids_len
        assert len(lm_label_ids) == input_ids_len

        return input_ids, input_mask, segment_ids, lm_label_ids

    def _truncate_seq_pair(self, seq1, seq2):
        """Truncates a sequence pair in place to the maximum length."""
        max_len = self.seq_len - 3
        while len(seq1) + len(seq2) > max_len:
            if len(seq1) > len(seq2):
                seq1.pop()
            else:
                seq2.pop()

    def _do_lm_preproc(self, token_ids):
        output_label = []
        for i, token_id in enumerate(token_ids):
            prob = random.random()
            # mask token with 15% probability
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    token_ids[i] = self.mask_token_id

                # 10% randomly change token to random token
                elif prob < 0.9:
                    token_ids[i] = \
                        random.randint(0, self.tokenizer_num_tokens - 1)

                # -> rest 10% randomly keep current token

                # append current token to output (we will predict these later)
                output_label.append(token_id)

            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-1)

        return token_ids, output_label


def bert_dataset_collate_fn(samples):
    seq_lens = [len(s[0]) for s in samples]
    max_seq_len = max(seq_lens)

    batch_input_ids = []
    batch_input_mask = []
    batch_segment_ids = []
    batch_lm_label_ids = []
    batch_is_next_label = []

    for seq_len, sample in zip(seq_lens, samples):
        if seq_len < max_seq_len:
            padding = torch.zeros(max_seq_len - seq_len, dtype=torch.long)
            batch_input_ids.append(torch.cat([sample[0], padding]))
            batch_input_mask.append(torch.cat([sample[1], padding]))
            batch_segment_ids.append(torch.cat([sample[2], padding]))
            batch_lm_label_ids.append(torch.cat([sample[3], padding]))
        else:
            batch_input_ids.append(sample[0])
            batch_input_mask.append(sample[1])
            batch_segment_ids.append(sample[2])
            batch_lm_label_ids.append(sample[3])
        batch_is_next_label.append(sample[4])

    return (
        torch.stack(batch_input_ids, dim=0),
        torch.stack(batch_input_mask, dim=0),
        torch.stack(batch_segment_ids, dim=0),
        torch.stack(batch_lm_label_ids, dim=0),
        torch.stack(batch_is_next_label, dim=0)
    )


def parse_args():
    parser = argparse.ArgumentParser()

    # I/O
    parser.add_argument(
        'checkpoint_dir', metavar='CHECKPOINT_DIR', type=str,
        help='The directory to store all training artifacts and outputs')
    parser.add_argument(
        'train_corpus', metavar='TRAIN_CORPUS', type=str,
        help='The input train corpus as required by '
        'https://github.com/huggingface/pytorch-pretrained-BERT')
    parser.add_argument(
        'spm_model', metavar='SPM_MODEL', type=str,
        help='The path to the sentencepiece model')
    parser.add_argument(
        'bert_config', metavar='BERT_CONFIG', type=str,
        help='The path to the bert config file')

    # Data loader configs
    data_group = parser.add_argument_group('data')
    data_group.add_argument(
        '--on_memory', action='store_true',
        help='Should training corpus be stored on memory?')
    data_group.add_argument(
        '--num_workers', type=int, default=0,
        help='How many workers should be used to load data with? Default 0. '
        '0 means all data loading is done on main thread')

    # Inference configs
    infer_group = parser.add_argument_group('infer')
    infer_group.add_argument(
        '--eff_batch_size', type=int, default=32,
        help='The effective batch size to use. Actual batch size will be '
        'changed by gradient_accumulation_steps')
    infer_group.add_argument(
        '--keep_punc', action='store_true',
        help='Should punctuations be kept?')
    # infer_group.add_argument(
    #     '--fp16', action='store_true',
    #     help='Should 16-bit fp be used over 32-bit fp?')

    # Training configs
    train_group = parser.add_argument_group('train')
    train_group.add_argument(
        '--eff_num_iters', type=int, default=10000,
        help='The effective number of steps to train for. Will be split '
        'between multiple GPUs')
    train_group.add_argument(
        '--gradient_accumulation_steps', type=int, default=1,
        help='The number of update steps to accumulate before performing a '
        'backward/update pass')
    train_group.add_argument(
        '--learning_rate', type=float, default=3e-5,
        help='The initial learning rate for Adam')
    train_group.add_argument(
        '--warmup_proportion', type=float, default=0.1,
        help='The proportion of training to perform linear learning rate '
        'warmup.')
    # train_group.add_argument(
    #     '--loss_scale', type=float, default=0,
    #     help='Loss scaling to improve fp16 numerical stability. Only used '
    #     'when fp16 is set to true. '
    #     '0 (default value): dynamic loss scaling. '
    #     'Positive power of 2: static loss scaling value.')

    # Other configs
    other_group = parser.add_argument_group('other')
    other_group.add_argument(
        '--no_cuda', action='store_true',
        help='Should GPU not be used?')
    other_group.add_argument(
        '--seed', type=int, default=42,
        help='random seed for initialization')
    other_group.add_argument(
        '--local_rank', type=int, default=-1,
        help='For distributed training')

    return parser.parse_args()


def check_args(args):
    assert args.gradient_accumulation_steps >= 1, (
        'Invalid gradient_accumulation_steps parameter: {}, '
        'should be >= 1'.format(args.gradient_accumulation_steps)
    )
    return args


def makedirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def set_random_seed(seed, is_distributed_training=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if is_distributed_training:
        torch.cuda.manual_seed_all(seed)


def get_or_create_model(
    bert_config,
    device,
    is_distributed_training=False,
    num_iters=10000,
    learning_rate=3e-5,
    warmup_proportion=0.1
):
    # Prepare model
    model = BertForPreTraining(bert_config)
    model.to(device)
    if is_distributed_training:
        # try:
        #     import apex as apm
        #     from apex.parallel import DistributedDataParallel as DDP
        # except ImportError:
        #     raise ImportError(
        #         "Please install apex from https://www.github.com/nvidia/apex"
        #         "to use distributed and fp16 training.")
        # model, optimizer = amp.initialize(model, optimizer, flags...)
        # model = DDP(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index])

    # Prepare optimizer params
    no_update_pattern = '(position_embeddings)'
    no_decay_pattern = '(bias)|(LayerNorm.bias)|(LayerNorm.weight)'
    no_update_params = []
    no_decay_params = []
    remaining_params = []
    for n, p in model.named_parameters():
        if re.findall(no_update_pattern, n):
            no_update_params.append(p)
        elif re.findall(no_decay_pattern, n):
            no_decay_params.append(p)
        else:
            remaining_params.append(p)

    optimizer_grouped_parameters = [
        {
            'params': no_update_params,
            'lr': 0,
            'weight_decay': 0.0
        }, {
            'params': no_decay_params,
            'lr': learning_rate,
            'weight_decay': 0.0
        }, {
            'params': remaining_params,
            'lr': learning_rate,
            'weight_decay': 0.0
        }
    ]

    # Prepare optimizer
    optimizer = BertAdam(
        optimizer_grouped_parameters,
        warmup=warmup_proportion,
        t_total=num_iters)

    return model, optimizer


def main():
    args = parse_args()
    check_args(args)
    is_main_process = args.local_rank <= 0

    # Determine if is main process and do logging
    if is_main_process:
        makedirs(args.checkpoint_dir)
        setup_logging(os.path.join(args.checkpoint_dir, 'train.log'), 'a')
        writer = SummaryWriter(args.checkpoint_dir)

        logger.info('Getting environmental info')
        logger.info(get_pretty_env_info() + '\n')

        logger.info('Getting script arguments')
        logger.info(args.__repr__() + '\n')

    # Set device
    is_distributed_training = False
    if args.no_cuda:
        device = torch.device('cpu')
    elif args.local_rank == -1:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        is_distributed_training = True
        world_size = torch.distributed.get_world_size()

    # Determine actual batch size and num training iterations to use
    batch_size = args.eff_batch_size // args.gradient_accumulation_steps
    num_iters = args.eff_num_iters
    if is_distributed_training:
        num_iters = num_iters // torch.distributed.get_world_size()

    # Set random seed
    # set_random_seed(args.seed, is_distributed_training)

    # Make tokenizer
    tokenizer = BertTokenizer(
        args.spm_model,
        sep_token=SEP_TOKEN,
        cls_token=CLS_TOKEN,
        mask_token=MASK_TOKEN,
        keep_punc=args.keep_punc)

    # Load bert configs from file
    bert_config = BertConfig.from_json_file(args.bert_config)
    bert_config.vocab_size = len(tokenizer.vocab)

    # Make dataset
    dataset = BertDataset(
        corpus_path=args.train_corpus,
        tokenizer=tokenizer,
        seq_len=bert_config.max_position_embeddings,
        on_memory=args.on_memory)

    # Make data_loader
    if is_distributed_training:
        sampler = torch.utils.data.DistributedSampler(
            dataset, rank=args.local_rank)
    else:
        sampler = torch.utils.data.RandomSampler(dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size
    )

    # Prepare model and optimizer
    model, optimizer = get_or_create_model(
        bert_config=bert_config,
        # prev_checkpoint
        device=device,
        is_distributed_training=is_distributed_training,
        num_iters=num_iters,
        learning_rate=args.learning_rate,
        warmup_proportion=args.warmup_proportion
    )
    model.train()

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)

    data_iter = data_loader.__iter__()
    next(data_iter)
    torch.distributed.barrier()

    if is_main_process:
        pbar = tqdm(total=num_iters, ncols=80)

    for step_nb in range(1, num_iters + 1):
        # Get next batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = data_loader.__iter__()
            batch = next(data_iter)
        if is_main_process:
            pbar.update(1)

        # Transfer batch to GPU
        input_ids, input_mask, segment_ids, lm_label_ids, is_next = \
            tuple(t.to(device, non_blocking=True) for t in batch)

        prediction_scores, seq_relationship_score = model(
            input_ids=input_ids,
            token_type_ids=segment_ids,
            attention_mask=input_mask
        )

        masked_lm_loss = loss_fn(
            prediction_scores.view(-1, bert_config.vocab_size),
            lm_label_ids.view(-1)
        )
        next_sentence_loss = loss_fn(
            seq_relationship_score.view(-1, 2),
            is_next.view(-1)
        )

        if is_distributed_training:
            all_losses = torch.stack([
                masked_lm_loss / world_size,
                next_sentence_loss / world_size
            ], dim=0)
            torch.distributed.reduce(all_losses, dst=0)
            masked_lm_loss = all_losses[0]
            next_sentence_loss = all_losses[1]
        loss = masked_lm_loss + next_sentence_loss

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss.backward()

        if (step_nb) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        if is_main_process and step_nb % 100 == 0:
            loss_dict = {
                'total_loss': loss.item() * args.gradient_accumulation_steps,
                'mlm_loss': masked_lm_loss.item(),
                'ns_loss': next_sentence_loss.item()
            }

            log_msg = 'step: {}'.format(step_nb)
            for k, v in loss_dict.items():
                log_msg += ' - {}: {:.2f}'.format(k, v)
                writer.add_scalar(k, v, global_step=step_nb)
            if len(log_msg) < 81:
                log_msg += ' ' * (80 - len(log_msg))
            sys.stdout.write('\r')
            sys.stdout.flush()
            logger.info(log_msg)

    if is_main_process:
        pbar.close()

    # Save a trained model
    if is_main_process:
        logger.info('Saving model')
        model_to_save = model.module if hasattr(model, 'module') else model
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)


if __name__ == "__main__":
    main()
