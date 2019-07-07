# -*- coding: utf-8 -*-

"""
Dataset for BERT pretraining
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import io
import random
from tqdm import tqdm

import torch
import torch.utils.data

__all__ = ['PretrainingBertDataset', 'bert_dataset_collate_fn']


class PretrainingBertDataset(torch.utils.data.Dataset):
    __initialized = False

    def __init__(
        self,
        corpus_path,
        tokenizer,
        seq_len,
        encoding='utf-8',
        on_memory=False,
        start_at_random_pos=False,
        pad_to_seq_len=True,
        sep_token='<sep>',
        cls_token='<cls>',
        mask_token='<mask>'
    ):
        self.tokenizer = tokenizer
        self.vocab_size = len(tokenizer.vocab)

        assert sep_token in tokenizer.vocab, \
            "'{}' is not an existing token in tokenizer".format(sep_token)
        assert cls_token in tokenizer.vocab, \
            "'{}' is not an existing token in tokenizer".format(cls_token)
        assert mask_token in tokenizer.vocab, \
            "'{}' is not an existing token in tokenizer".format(mask_token)
        self.sep_token_id = self.tokenizer.vocab[sep_token]
        self.cls_token_id = self.tokenizer.vocab[cls_token]
        self.mask_token_id = self.tokenizer.vocab[mask_token]

        assert isinstance(seq_len, int) and seq_len > 0, \
            'seq_len should be a positive integer, got {}'.format(seq_len)
        self.seq_len = seq_len
        self.pad_to_seq_len = bool(pad_to_seq_len)

        self.corpus_path = corpus_path
        self.encoding = encoding
        self.on_memory = bool(on_memory)
        self.start_at_random_pos = bool(start_at_random_pos)

        # Line buffer keeps the second sentence of a pair in memory and use as
        # first sentence in next pair
        self.line_buffer = None

        # Attributes to keep track of position in dataset
        self.current_doc = 0  # to avoid random sentence from same doc
        self.current_random_doc = 0

        # Load dataset
        self.corpus_lines = 0
        self.num_docs = 0
        if on_memory:
            # load samples into memory
            self._load_dataset_on_memory()
        else:
            # load dataset statistics lazily from disk
            self._load_dataset_off_memory()
        assert self.corpus_lines > 0, 'corpus is empty'
        assert self.num_docs > 0, 'corpus is empty'

    def _load_dataset_on_memory(self):
        self.sample_to_doc = []  # map sample index to doc and line
        self.all_docs = []
        with io.open(self.corpus_path, 'r', encoding=self.encoding) as f:
            doc = []
            for line in tqdm(f, desc='Loading Dataset'):
                line = line.strip()
                if line == '':
                    assert len(doc) >= 2, (
                        'Found a document with {} sentences. '
                        'All documents should have atleast 2 sentences'
                    ).fomat(len(doc))
                    self.all_docs.append(doc)
                    doc = []
                    # remove last added sample because there won't be a
                    # subsequent line anymore in the doc
                    self.sample_to_doc.pop()
                else:
                    # store as one sample
                    sample = {
                        'doc_id': len(self.all_docs),
                        'line': len(doc)
                    }
                    self.sample_to_doc.append(sample)
                    doc.append(line)
                    self.corpus_lines = self.corpus_lines + 1

        # if last row in file is not empty
        if self.all_docs[-1] != doc:
            self.all_docs.append(doc)
            self.sample_to_doc.pop()

        self.num_docs = len(self.all_docs)

    def _load_dataset_off_memory(self):
        with io.open(self.corpus_path, 'r', encoding=self.encoding) as f:
            doc_size = 0
            for line in tqdm(f, desc='Loading Dataset'):
                line = line.strip()
                if line == '':
                    self.num_docs += 1
                    assert doc_size >= 2, (
                        'Found a document with {} sentences. '
                        'All documents should have atleast 2 sentences'
                    ).fomat(doc_size)
                    doc_size = 0
                else:
                    self.corpus_lines += 1
                    doc_size += 1

            # if doc does not end with empty line
            if line != '':
                self.num_docs += 1

    def _initialize_dataset(self):
        """
        This method would open the files on the first call of __getitem__
        this way multiple workers can be used to load from this dataset

        Pretty hacky I know but atleast this would satisfy 99% of use cases
        """
        assert not self.__initialized, \
            'PretrainBertDataset has already been initialized'
        self.__initialized = True

        # Nothing is needed to be done if everything is already on memory
        if self.on_memory:
            return

        # Open files
        self.file = io.open(self.corpus_path, "r", encoding=self.encoding)
        self.random_file = \
            io.open(self.corpus_path, "r", encoding=self.encoding)

        # Flip random file to random doc
        start_random_doc = random.randrange(self.num_docs)
        desc = 'Opening self.random_file to random doc'
        pbar = tqdm(total=start_random_doc, desc=desc)
        while self.current_random_doc < start_random_doc:
            line = next(self.random_file).strip()
            if line == '':
                pbar.update(1)
                self.current_random_doc += 1
        pbar.close()
        assert self.current_random_doc == start_random_doc

        if self.start_at_random_pos:
            # Flip file to random doc
            start_doc = random.randrange(self.num_docs)
            desc = 'Opening self.file to random doc'
            pbar = tqdm(total=start_doc, desc=desc)
            while self.current_doc < start_doc:
                line = next(self.file).strip()
                if line == '':
                    pbar.update(1)
                    self.current_doc += 1
            pbar.close()
            assert self.current_doc == start_doc

    def get_next_line(self):
        """ Assumes not on_memory. Gets next line from self.file """
        try:
            line = next(self.file).strip()
            if line == '':
                self.current_doc += 1
                line = next(self.file).strip()
        except StopIteration:
            self.file.close()
            self.current_doc = 0
            self.file = io.open(self.corpus_path, "r", encoding=self.encoding)
            line = next(self.file).strip()
        return line

    def get_next_random_line(self):
        """ Assumes not on_memory. Gets next line from self.random_file """
        try:
            line = next(self.random_file).strip()
            if line == '':
                self.current_random_doc += 1
                line = next(self.random_file).strip()
        except StopIteration:
            self.random_file.close()
            self.current_random_doc = 0
            self.random_file = \
                io.open(self.corpus_path, "r", encoding=self.encoding)
            line = next(self.random_file).strip()
        return line

    def get_corpus_pair(self, idx):
        """
        Get one sample from corpus consisting of a pair of two subsequent lines
        from the same doc.
        :param index: int, index of sample.
        :return: (str, str), two subsequent sentences from corpus
        """
        if self.on_memory:
            sample_doc_id = self.sample_to_doc[idx]['doc_id']
            sample_line_nb = self.sample_to_doc[idx]['line']
            t1 = self.all_docs[sample_doc_id][sample_line_nb]
            t2 = self.all_docs[sample_doc_id][sample_line_nb + 1]

            # used later to avoid random nextSentence from same doc
            self.current_doc = sample_doc_id

        else:
            if self.line_buffer is None:
                t1 = self.get_next_line()
                t2 = self.get_next_line()

            else:
                # use t2 from previous iteration as new t1
                t1 = self.line_buffer
                t1_doc_nb = self.current_doc
                t2 = self.get_next_line()
                t2_doc_nb = self.current_doc

                while t1_doc_nb != t2_doc_nb:
                    t1 = self.get_next_line()
                    t1_doc_nb = self.current_doc
                    t2 = self.get_next_line()
                    t2_doc_nb = self.current_doc

                self.line_buffer = t2

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
                rand_doc_idx = random.randrange(self.all_docs)
                rand_doc = self.all_docs[rand_doc_idx]
                line = rand_doc[random.randrange(len(rand_doc))]
                self.current_random_doc = rand_doc
            else:
                rand_index = random.randrange(min(self.corpus_lines, 10000))
                rand_index += 1
                for _ in range(rand_index):
                    line = self.get_next_random_line()
            # check if our picked random line is really from another doc like
            # we want it to be
            if self.current_random_doc != self.current_doc:
                break
        return line

    def generate_new_sample(self, idx):
        """
        Get one sample from corpus consisting of two sentences. With prob. 50%
        these are two subsequent sentences from one doc. With 50% the second
        sentence will be a random one from another doc.
        :param index: int, index of sample.
        :return: (str, str, int), sentence 1, sentence 2, isNextSentence Label
        """
        t1, t2 = self.get_corpus_pair(idx)
        if random.random() > 0.5:
            label = 0
        else:
            t2 = self.get_random_line()
            label = 1
        # This is purely to catch unexpected errors
        assert len(t1) > 0 and len(t2) > 0
        return t1, t2, label

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
                    token_ids[i] = random.randrange(self.vocab_size)

                # -> rest 10% randomly keep current token

                # append current token to output (we will predict these later)
                output_label.append(token_id)

            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-1)

        return token_ids, output_label

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

    def __len__(self):
        # Last line of each doc would not be used
        return self.corpus_lines - self.num_docs

    def __getitem__(self, idx):
        if not self.__initialized:
            self._initialize_dataset()

        t1, t2, is_next_label = self.generate_new_sample(idx)
        input_ids, input_mask, segment_ids, lm_label_ids = \
            self.convert_sample_to_features(t1, t2)

        return (
            torch.tensor(input_ids),
            torch.tensor(input_mask),
            torch.tensor(segment_ids),
            torch.tensor(lm_label_ids),
            torch.tensor(is_next_label)
        )


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
