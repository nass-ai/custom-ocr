#!/usr/bin/env python3
# -*- coding: utf-8 -*-


vocab_file="./vocabulary.txt"

class TextFeaturizer(object):
  """Extract text feature based on char-level granularity.

  By looking up the vocabulary table, each input string (one line of transcript)
  will be converted to a sequence of integer indexes.
  """

  def __init__(self, vocab_file):
    lines = []

    f= open(vocab_file, "r")
    lines.extend(f.readlines())
    f.close()

    self.token_to_index = {}
    self.index_to_token = {}
    self.speech_labels = ""
    index = 0


    for line in lines:
      line = line[:-1]  # Strip the '\n' char.

      self.token_to_index[line] = index
      self.index_to_token[index] = line
      self.speech_labels += line
      index += 1

    self.token_to_index.update({"\n":index})
    self.index_to_token.update({index:"\n"})
    

  def encode_sents(self, sents):
      """
      convert sentences<list(lists)> to numerical encodings
      """
      tok2idx = self.token_to_index
      #print(tok2idx)
      encoded= []
      for sent in sents:
          txt_inside=sent[0]
          enc = [tok2idx[ch] for ch in txt_inside]
          encoded.append(enc)

      return encoded



  def decode_sents(self, enc_sents):
      """
      convert numerical encoded predictions<tf.Tensor> to sentences
      """
      decoded_sents="".join(self.index_to_token[i] for i in enc_sents)
      return decoded_sents
