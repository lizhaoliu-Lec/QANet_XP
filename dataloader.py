# -*- coding:utf8 -*-

import json
import logging
import numpy as np
from collections import Counter
import jieba


def word_tokenize(sent):
    if isinstance(sent, list):
        tokens = sent
    else:
        tokens = jieba.lcut(sent)
    return [token for token in tokens if len(token) >= 1]


class DataLoader(object):
    """
    This module implements the APIs for loading and using baidu reading comprehension dataset
    """

    def __init__(self, max_a_len, max_p_len, max_q_len, max_char_len,
                 train_files=[], dev_files=[], test_files=[]):
        self.logger = logging.getLogger("brc")
        self.max_a_len = max_a_len
        self.max_p_len = max_p_len
        self.max_q_len = max_q_len
        self.max_char_len = max_char_len

        self.train_set, self.dev_set, self.test_set = [], [], []
        if train_files:
            for train_file in train_files:
                self.logger.info('---train file-----{}'.format(train_file))
                self.train_set += self._load_dataset(train_file, train=True)
            self.logger.info('Train set size: {} questions.'.format(len(self.train_set)))

        if dev_files:
            for dev_file in dev_files:
                self.logger.info('---dev file-----{}'.format(dev_file))
                self.dev_set += self._load_dataset(dev_file, train=True)
            self.logger.info('Dev set size: {} questions.'.format(len(self.dev_set)))

        if test_files:
            for test_file in test_files:
                self.test_set += self._load_dataset(test_file)
            self.logger.info('Test set size: {} questions.'.format(len(self.test_set)))

    def _load_dataset(self, data_path, train=False):
        """
        Loads the dataset
        Args:
            data_path: the data file to load
        """
        max_char_num = 0
        with open(data_path, encoding='UTF-8') as fin:
            data_set = []
            for idx, line in enumerate(fin):

                sample = json.loads(line.strip())

                if len(sample['answer']) > self.max_a_len:
                    print(sample)
                    print('got answer idx `%d` bigger than max_a_len `%d`, ignore it.' % (
                        sample['answer'][0], self.max_a_len))
                    continue

                question_tokens = word_tokenize(sample['question'])
                sample['question_tokens'] = question_tokens
                question_chars = [list(token) for token in question_tokens]
                sample['question_chars'] = question_chars

                for char in question_chars:
                    if len(char) > max_char_num:
                        max_char_num = len(char)

                context_tokens = word_tokenize(sample['context'])
                sample['context_tokens'] = context_tokens
                context_chars = [list(token) for token in context_tokens]
                sample['context_chars'] = context_chars

                for char in context_chars:
                    if len(char) > max_char_num:
                        max_char_num = len(char)

                data_set.append(sample)
        return data_set

    def _one_mini_batch(self, data, indices, pad_id, pad_char_id):
        """
        Get one mini batch
        Args:
            data: all data
            indices: the indices of the samples to be selected
            pad_id:
        Returns:
            one batch of data
        """
        batch_data = {
            'raw_data': [data[i] for i in indices],
            'question_token_ids': [],
            'question_char_ids': [],
            'question_length': [],
            'context_token_ids': [],
            'context_length': [],
            'context_char_ids': [],
            'label': [],
        }

        for sidx, sample in enumerate(batch_data['raw_data']):
            batch_data['question_token_ids'].append(sample['question_token_ids'])
            batch_data['question_char_ids'].append(sample['question_char_ids'])
            batch_data['question_length'].append(len(sample['question_token_ids']))
            batch_data['context_token_ids'].append(sample['context_token_ids'])
            batch_data['context_length'].append(min(len(sample['context_token_ids']), self.max_p_len))
            batch_data['context_char_ids'].append(sample['context_char_ids'])

        batch_data, padded_p_len, padded_q_len = self._dynamic_padding(batch_data, pad_id, pad_char_id)
        for sample in batch_data['raw_data']:
            batch_data['label'].append(sample['answer'][0])
        return batch_data

    def _dynamic_padding(self, batch_data, pad_id, pad_char_id):
        """
        Dynamically pads the batch_data with pad_id
        """
        pad_char_len = self.max_char_len
        pad_p_len = self.max_p_len  # min(self.max_p_len, max(batch_data['passage_length']))
        pad_q_len = self.max_q_len  # min(self.max_q_len, max(batch_data['question_length']))
        batch_data['context_token_ids'] = [(ids + [pad_id] * (pad_p_len - len(ids)))[: pad_p_len]
                                           for ids in batch_data['context_token_ids']]
        for index, char_list in enumerate(batch_data['context_char_ids']):
            # print(batch_data['passage_char_ids'])
            for char_index in range(len(char_list)):
                if len(char_list[char_index]) >= pad_char_len:
                    char_list[char_index] = char_list[char_index][:self.max_char_len]
                else:
                    char_list[char_index] += [pad_char_id] * (pad_char_len - len(char_list[char_index]))
            batch_data['context_char_ids'][index] = char_list
        batch_data['context_char_ids'] = [(ids + [[pad_char_id] * pad_char_len] * (pad_p_len - len(ids)))[:pad_p_len]
                                          for ids in batch_data['context_char_ids']]

        # print(np.array(batch_data['passage_char_ids']).shape, "==========")

        batch_data['question_token_ids'] = [(ids + [pad_id] * (pad_q_len - len(ids)))[: pad_q_len]
                                            for ids in batch_data['question_token_ids']]
        for index, char_list in enumerate(batch_data['question_char_ids']):
            for char_index in range(len(char_list)):
                if len(char_list[char_index]) >= pad_char_len:
                    char_list[char_index] = char_list[char_index][:self.max_char_len]
                else:
                    char_list[char_index] += [pad_char_id] * (pad_char_len - len(char_list[char_index]))
            batch_data['question_char_ids'][index] = char_list
        batch_data['question_char_ids'] = [(ids + [[pad_char_id] * pad_char_len] * (pad_q_len - len(ids)))[:pad_q_len]
                                           for ids in batch_data['question_char_ids']]

        return batch_data, pad_p_len, pad_q_len

    def word_iter(self, set_name=None):
        """
        Iterates over all the words in the dataset
        Args:
            set_name: if it is set, then the specific set will be used
        Returns:
            a generator
        """
        if set_name is None:
            data_set = self.train_set + self.dev_set + self.test_set
        elif set_name == 'train':
            data_set = self.train_set
        elif set_name == 'dev':
            data_set = self.dev_set
        elif set_name == 'test':
            data_set = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        if data_set is not None:
            for sample in data_set:
                for token in sample['question_tokens']:
                    yield token
                for token in sample['context_tokens']:
                    yield token

    def convert_to_ids(self, vocab):
        """
        Convert the question and paragraph in the original dataset to ids
        Args:
            vocab: the vocabulary on this dataset
        """
        for data_set in [self.train_set, self.dev_set, self.test_set]:
            if data_set is None:
                continue
            for sample in data_set:
                sample['question_token_ids'] = vocab.convert_word_to_ids(sample['question_tokens'])
                sample["question_char_ids"] = vocab.convert_char_to_ids(sample['question_tokens'])
                sample['context_token_ids'] = vocab.convert_word_to_ids(sample['context_tokens'])
                sample["context_char_ids"] = vocab.convert_char_to_ids(sample['context_tokens'])

    def next_batch(self, set_name, batch_size, pad_id, pad_char_id, shuffle=True):
        """
        Generate data batches for a specific dataset (train/dev/test)
        Args:
            set_name: train/dev/test to indicate the set
            batch_size: number of samples in one batch
            pad_id: pad id
            pad_char_id: pad char id
            shuffle: if set to be true, the data is shuffled.
        Returns:
            a generator for all batches
        """
        if set_name == 'train':
            data = self.train_set
        elif set_name == 'dev':
            data = self.dev_set
        elif set_name == 'test':
            data = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        data_size = len(data)
        indices = np.arange(data_size)
        if shuffle:
            np.random.shuffle(indices)
        for batch_start in np.arange(0, data_size, batch_size):
            batch_indices = indices[batch_start: batch_start + batch_size]
            yield self._one_mini_batch(data, batch_indices, pad_id, pad_char_id)
