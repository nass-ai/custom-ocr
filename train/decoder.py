#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 00:11:55 2020
"""

from nltk.metrics import distance
import tensorflow as tf
tf.get_logger().setLevel('INFO')

class Decoder(object):

    def __init__(self, labels, blank_index=89, decoding="greedy", beam_width=None):
        """Decoder initialization.
    
        Arguments:
          labels: a string specifying the speech labels(character set) for the decoder to use.
          blank_index: an integer specifying index for the blank character.
          Defaults to 89.
        """
        self.labels=labels
        self.blank_index=blank_index
        self.decoding=decoding
        self.int_to_char = dict([(i, ch) for (i, ch) in enumerate(labels)])

    def seq_to_string(self, sequence):
        """convert number encoded predictions to string."""
        return "".join(self.int_to_char[i]  for i in sequence)

    def cer(self, decode, target):
        """Computes the Character Error Rate (CER).

        CER is defined as the edit distance between the two given strings.
    
        Args:
          decode: a string of the decoded output.
          target: a string for the ground truth label.
    
        Returns:
          A float number denoting the CER for the current sentence pair.
        """
        return distance.edit_distance(decode, target)

    def wer(self, decode, target):
        """Computes the Word Error Rate (WER).

        WER is defined as the edit distance between the two provided sentences after
        tokenizing to words.
    
        Args:
          decode: string of the decoded output.
          target: a string for the ground truth label.
    
        Returns:
          A float number for the WER of the current decode-target pair.
        """
        #print(pred_seq, gt_seq)
        words = set(decode.split() + target.split())
        words2char = dict([(word, str(i)) for (i, word) in enumerate(words)])

        new_decode = [words2char[word] for word in decode.split()]
        new_target = [words2char[word] for word in target.split()]

        error = distance.edit_distance(''.join(new_decode), ''.join(new_target))
        return error

    def decode(self, logits):
        """
        Decode the best guess from logits according
        to the decoding mechanism (self.decoding: greedy or beam) , default=greedy

        Inputs: logits <tf.Tensor>(max_seq_length, batch_size, num_classes)
        output: transcript<tf.Tensor>(batch_size, max_seq_length, dtype=tf.str)

        """
        max_seq_len, batch_size, num_classes = logits.shape


        if self.decoding=="beam":
            self.beam_width=10
            decoded, log_probs = tf.compat.v1.nn.ctc_beam_search_decoder(inputs=logits,
                                                               sequence_length=[max_seq_len]*batch_size,
                                                               beam_width=self.beam_width)

            # the gold is in the zeroth item ...weeew
            # which is a sparse matrix, <indices, values, dense_shape >
            return decoded[0]

        elif self.decoding=="greedy":
            decoded, log_probs = tf.compat.v1.nn.ctc_greedy_decoder(inputs=logits,
                                                          sequence_length=[max_seq_len]*batch_size)
            # the gold is in the zeroth item ...weeew
            # which is a sparse matrix, <indices, vallues, dense_shape>
            return decoded[0]
        else:
            raise ValueError("choose either 'greedy' or 'beam' as the decoder in the model definition!")
