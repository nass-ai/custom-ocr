#%%writefile train/train.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 11:47:55 2020
"""
from decoder import Decoder
from feat_ext import TextFeaturizer
from model import Model
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm

import datetime
import os
import pandas as pd
import tensorflow as tf


# 10, 442

# change this later on
data_dir = "../dataset/data.csv"
VOCAB_FILE ="./vocabulary.txt"

df = pd.read_csv(data_dir, encoding="utf-8")

featurizer = TextFeaturizer(VOCAB_FILE)

global losses, batch_size
batch_size=16
losses=[]


def fetch_data(n_train, n_val, batch_size=batch_size):
    """
    Read text line images from their file paths as stored in a csv file (../dataset/data.csv); pair them with their ground truth TRANSCRIPTS
    Input: n_train <int> : number of training num_of_examples
               n_val <int> : number of validation data points
               batch_size : size of minibatch

    Returns: tf.Data generator.
    """

    # read out paths to images and their groundtruths from a CSV file
    
    train_images_paths = [str(x) for x  in list(df["FILE NAMES"].iloc[:n_train])]
    train_transcripts = [[str(x)] for x in list(df["TRANSCRIPTS"].iloc[:n_train].fillna(' '))]

    # indexing pandas objects beyond bounds return empty list so I just use [n_train:]
    val_images_paths = [str(x) for x in list(df["FILE NAMES"].iloc[n_train : n_train+n_val])]
    val_transcripts = [[str(x)] for x in list(df["TRANSCRIPTS"].iloc[n_train: n_train+n_val].fillna(' '))]

    train_images = list(map(parse_fn, train_images_paths))
    val_images =list( map(parse_fn, val_images_paths))


    train_labels, train_seq_lengths = compute_label_features(train_transcripts)
    val_labels, val_seq_lengths= compute_label_features(val_transcripts)

    train_data_gen = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(train_images),
                                                       train_labels, train_transcripts))
    val_data_gen = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(val_images),
                                                   val_labels, val_transcripts))

    return (train_data_gen, val_data_gen, train_seq_lengths, val_seq_lengths)

def parse_fn(fname, image_size=(1024, 64)):
    """
    Read resize and rescale text line images
    Input: fname <str> path to the image file
           image_size: pixel dimension of the rescaled image
    Returns: preprocessed image
    """
    new_w, new_h = image_size[0], image_size[1]
    im_obj=tf.io.read_file(fname)
    im2nums = tf.io.decode_png(im_obj)

    im2nums = tf.cast(im2nums, tf.float32)/255.0

    H, W, B = im2nums.shape
    im2nums = tf.convert_to_tensor(im2nums, dtype=tf.float32)

    im2nums = tf.reshape(im2nums, [ W, H, 1])
    img = tf.image.resize(im2nums, (new_h, new_w))


    return img

def compute_label_features(text_batch):
    """
    Encode transcripts and return lengths of individual transcripts
    per time step
    Input: text <list<list>> A batch of transcripts
    Returns: encoded text <list<list>>
                   seq_lengths <list> length of of the encoded ground transcripts
                   i.e the number of tokens in each text line image
    """

    encoded_text=featurizer.encode_sents(text_batch)
    seq_lengths = [len(token) for token in encoded_text]
    encoded_text=pad_sequence(encoded_text)
    encoded_text = tf.convert_to_tensor(encoded_text)

    return encoded_text, seq_lengths

def pad_sequence(sequence, pad_token=45):
    """
    Pad list of sequences according to the longest sequence in the batch.
    Input: sequence (list[list[int]]): list of sequences, where each sentence
                                    is represented as a list of words
              pad_token (int): padding token
    Returns: seq_padded (list[list[int]]): list of sequences where sequences shorter
                    than the max length sequences are padded out with the pad_token, such that
                    each sentences in the batch now has equal length.
    Output shape: (batch_size, max_sentence_length)
    """
    max_len = max(len(sent) for sent in sequence)


    seq_padded =[]
    for sent in sequence:
        if len(sent) < max_len:
            # space has index
            sent.extend([pad_token]*(max_len - len(sent)))
        seq_padded.append(sent)

    return seq_padded

def main(epochs=3):
    """
    Call the OCR model class, define the architecture; run the training loop
    """
    print("entered main already")
    global losses, batch_size
    batch_size=4
    losses=[]


    f = open(VOCAB_FILE, "r")
    vocab = "".join([ch[:-1] for ch in f.readlines()])
    f.close()
    num_classes = len(vocab)


    train_data,  val_data, train_seq_lengths, val_seq_lengths = fetch_data(n_train=128,
                                                                           n_val = 64,
                                                                           batch_size=batch_size)
    train_data =list(train_data.as_numpy_iterator())
    val_data = list(val_data.as_numpy_iterator())

    X_train= tf.stack([x[0] for x in train_data])
    y_train = tf.stack([y[1] for y in train_data])
    train_transcripts = [z[2] for  z in train_data]


    X_val= tf.stack([x[0] for x in val_data])
    y_val = tf.stack([y[1] for y in val_data])
    val_transcripts = [z[2] for  z in val_data]

    # split the training data into batches and deal with the labels during ctc
    n_train=len(train_data)
    X_tr_batches =tf.split(X_train, n_train//batch_size)
    y_tr_batches = tf.split(y_train, n_train//batch_size)
    print(len(X_tr_batches), len(y_tr_batches))
    gT_tr_batches = tf.split(train_transcripts,n_train//batch_size)

    y_tr_seq_lengths_batches = tf.split(tf.constant(train_seq_lengths), n_train//batch_size)
    y_val_seq_lengths_batches = tf.split(tf.constant(val_seq_lengths), n_train//batch_size)
    gT_val_batches = tf.split(val_transcripts, n_train//batch_size)

    model = Model(num_cnn_layers=11,
                  num_rnn_layers=9,
                  rnn_type="gru",
                  is_bidirectional=True,
                  rnn_hidden_size=256,
                  num_classes=num_classes,
                  is_dropout=True,
                  use_bias=True)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)


    # set up tf.summary to keep track of losses  and accuracies ...for tensorboard
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)


    # save utilities
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=3)

    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
      print("Restored from {}".format(manager.latest_checkpoint))
    else:
      print("Initializing from scratch.")
    #checkpoint_dir = './training_checkpoints'
    #checkpoint_prefix=os.path.join(checkpoint_dir, "ckpt")
    #checkpoint = tf.train.Checkpoint(optimizer=optimizer,
    #                                 model = model)

    # restoring the latest checkpoint in checkpoint_dir
    #checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    # Training Loop
    for epoch in tqdm(range(epochs)):
        print()
        print()
        print()
        print()
        print("epoch", epoch+1)
        print(batch_size)
        for batch in range(n_train//batch_size):
            batch_input = (X_tr_batches[batch],y_tr_batches[batch])
            train_seq_lengths = y_tr_seq_lengths_batches[batch]
            gT_transcripts=gT_tr_batches[batch]

            X_train,y_train = batch_input
            # conform
            y_train=tf.convert_to_tensor(y_train)
            with tf.GradientTape() as tape:
                # call the nn upon the images
                logits = model(X_train, training=True)
                # compute the length
                batch_size, max_seq_len, num_classes=logits.shape
                # logits has shape B,T,C ...reshape to T,B,C to make tf.nn.ctc_loss happy
                logits = tf.transpose(logits, [1,0,2])
                # convert dense labels to a sparse matrix for ctc_loss
                #sparse_labels = tf.keras.backend.ctc_label_dense_to_sparse(y_train,
                #                                                           label_lengths=train_seq_lengths)
                loss_tensor = tf.reduce_mean(tf.nn.ctc_loss(labels=y_train, logits=logits,
                                                            label_length=train_seq_lengths,logit_length=[max_seq_len]*batch_size,
                                                            logits_time_major=True, blank_index=89))

                print("loss", loss_tensor)
                # decoding
                decoder  = Decoder(labels=vocab,decoding="beam", beam_width=100)
                sparse_dec_seq = decoder.decode(logits)
                labels = tf.sparse.to_dense(sparse_dec_seq)

                total_wer, total_cer= 0, 0

                for i in range(batch_size):

                    dec_text = featurizer.decode_sents(tf.slice(labels, [i,0], [1, labels.shape[1]]).numpy().tolist()[0])
                    gT_transcript = gT_transcripts[i].numpy().tolist()[0].decode("utf-8")
                    #if i==10:
                    print("{}, decoded-{}, target-{}".format(i, dec_text, gT_transcript))
                    #print()
                    total_cer += decoder.cer(dec_text,gT_transcript)/float(len(gT_transcript)+1)
                    total_wer += decoder.wer(dec_text, gT_transcript)/float(len(gT_transcript.split())+1)

                # get mean value
                print(total_cer, total_wer)
                total_cer /= batch_size
                total_wer /= batch_size


                # evaluate

                losses.append((loss_tensor, total_cer))
                print("loss: {:.2f}; CER: {:.2f}; WER: {:.2f}".format(loss_tensor,total_cer, total_wer))

                grads = tape.gradient(loss_tensor, model.trainable_variables)

                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                ckpt.step.assign_add(1)
                if int(ckpt.step) % 10 == 0:
                  save_path = manager.save()
                  print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', loss_tensor, step=epoch)
            tf.summary.scalar('CER', total_cer, step=epoch)
            tf.summary.scalar('WER', total_wer, step=epoch)


if __name__=="__main__":
    main()
    #%tensorboard --logdir logs/gradient_tape

