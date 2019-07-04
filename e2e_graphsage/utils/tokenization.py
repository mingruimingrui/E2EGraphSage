"""
Taken from https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/pytorch_pretrained_bert/tokenization.py
Similar in every aspect but uses sentencepiece as word tokenizer to account for
score
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import warnings
import collections

import unicodedata
from six import binary_type

import sentencepiece as spm

VOCAB_NAME = 'vocab.txt'


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def _clean_text(text):
    """Performs invalid character removal and whitespace cleanup on text."""
    output = []
    for char in text:
        cp = ord(char)
        if cp == 0 or cp == 0xfffd or _is_control(char):
            continue
        if _is_whitespace(char):
            output.append(" ")
        else:
            output.append(char)
    return "".join(output)


def _strip_accents(text):
    """Strips accents from a piece of text."""
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat == "Mn":
            continue
        output.append(char)
    return "".join(output)


def _cast_to_unicode(text):
    if isinstance(text, binary_type):
        text = text.decode('utf-8', errors='ignore')
    return text


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


def _is_chinese_char(cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    return ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F))


class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(
        self,
        do_lower_case=True,
        unk_token='<unk>',
        sep_token='<sep>',
        pad_token='<pad>',
        cls_token='<cls>',
        mask_token='<mask>',
        never_split=('<unk>', '<sep>', '<pad>', '<cls>', '<mask>'),
        keep_punc=True
    ):
        """Constructs a BasicTokenizer.
        Args:
            do_lower_case: Should text be cast to lowercase?
            keep_punc: Should punctuations be kept?
                Characters such as "^", "$", and "`" are not in the Unicode
                Punctuation class but we treat them as punctuation anyways
        """
        self.do_lower_case = do_lower_case
        self.unk_token = unk_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.mask_token = mask_token
        self.never_split = set(never_split)
        self.keep_punc = keep_punc

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = _cast_to_unicode(text)
        text = _clean_text(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case and token not in self.never_split:
                token = token.lower()
                token = _strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))
        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        if text in self.never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                if self.keep_punc:
                    output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]


class BertTokenizer(object):
    """Runs end-to-end tokenization: punctuation splitting + wordpiece"""

    def __init__(
        self,
        spm_model_file,
        do_lower_case=True,
        max_len=None,
        do_basic_tokenize=True,
        unk_token='<unk>',
        sep_token='<sep>',
        pad_token='<pad>',
        cls_token='<cls>',
        mask_token='<mask>',
        never_split=('<unk>', '<sep>', '<pad>', '<cls>', '<mask>'),
        keep_punc=True
    ):
        """Constructs a BertTokenizer.
        Args:
            vocab_file: Path to a one-wordpiece-per-line vocabulary file
            do_lower_case: Whether to lower case the input
                Only has an effect when do_wordpiece_only=False
            do_basic_tokenize: Whether to do basic tokenization before
                wordpiece.
            max_len: An artificial maximum length to truncate tokenized
                sequences to;
                Effective maximum length is always the minimum of this
                value (if specified) and the underlying BERT model's
                sequence length.
            never_split: List of tokens which will never be split during
                tokenization.
                Only has an effect when do_wordpiece_only=False
            keep_punc: Should punctuations be kept?
                Characters such as "^", "$", and "`" are not in the Unicode
                Punctuation class but we treat them as punctuation anyways
        """
        if not os.path.isfile(spm_model_file):
            err_msg = "Can't find a spm model file at path '{}'"
            err_msg = err_msg.format(spm_model_file)
            raise ValueError(err_msg)

        self.spm_model = spm.SentencePieceProcessor()
        self.spm_model.Load(spm_model_file)

        # Extract ids_to_tokens and vocab for compatibility
        all_tokens = []
        for i in range(len(self.spm_model)):
            all_tokens.append(self.spm_model.IdToPiece(i))
        all_tokens = list(map(_cast_to_unicode, all_tokens))

        self.vocab = collections.OrderedDict(zip(
            all_tokens,
            range(len(all_tokens))
        ))
        self.ids_to_tokens = collections.OrderedDict(zip(
            range(len(all_tokens)),
            all_tokens
        ))

        self.do_basic_tokenize = do_basic_tokenize
        if do_basic_tokenize:
            self.basic_tokenier = BasicTokenizer(
                do_lower_case=do_lower_case,
                unk_token=unk_token,
                sep_token=sep_token,
                pad_token=pad_token,
                cls_token=cls_token,
                mask_token=mask_token,
                never_split=never_split,
                keep_punc=keep_punc
            )
        self.max_len = max_len if max_len is not None else int(1e12)

    def tokenize(self, text):
        text = _cast_to_unicode(text)
        split_tokens = []
        if self.do_basic_tokenize:
            for token in self.basic_tokenier.tokenize(text):
                for sub_token in self.spm_model.EncodeAsPieces(token):
                    split_tokens.append(_cast_to_unicode(sub_token))
        else:
            split_tokens = self.spm_model.EncodeAsPieces(text)
            split_tokens = list(map(_cast_to_unicode, split_tokens))
        return split_tokens

    def tokenize_as_ids(self, text):
        text = _cast_to_unicode(text)
        split_token_ids = []
        if self.do_basic_tokenize:
            for token in self.basic_tokenier.tokenize(text):
                for sub_token_id in self.spm_model.EncodeAsIds(token):
                    split_token_ids.append(sub_token_id)
        else:
            split_token_ids = self.spm_model.EncodeAsIds(text)
        return split_token_ids

    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            ids.append(self.vocab[token])
        if len(ids) > self.max_len:
            warnings.warn(
                "Token indices sequence length is longer than the specified "
                "maximum sequence length for this BERT model ({} > {}). "
                "Running this sequence through BERT will result in indexing "
                "errors".format(len(ids), self.max_len)
            )
        return ids

    def convert_ids_to_tokens(self, ids):
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return tokens

    def save_vocabulary(self, vocab_path):
        raise Exception(
            'This function exists in the original BertTokenizer '
            'but is not needed here.')
