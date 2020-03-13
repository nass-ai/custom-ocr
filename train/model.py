#%%writefile ./train/model.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 13:25:28 2020
"""
import tensorflow as tf
# suppress warnings from tensorflow
#tf.get_logger().setLevel(logging.ERROR)
"""
NN definitions
"""

RNN_FLAVORS = {"lstm": tf.compat.v1.nn.rnn_cell.BasicLSTMCell,
                        "gru": tf.compat.v1.nn.rnn_cell.GRUCell,
                        "rnn": tf.compat.v1.nn.rnn_cell.RNNCell}

# Total number of output filters for convolution operation
CONV_FILTERS=93

def cnn_layer(inputs, filters,  kernel_size, strides, layer_id, is_dropout):
    """Defines 2D convolutional.

      Args:
        inputs: input data for convolution layer.
        filters: an integer, number of output filters in the convolution.
        kernel_size: a tuple specifying the height and width of the 2D convolution
          window.
        strides: a tuple specifying the stride length of the convolution.
        layer_id: an integer specifying the layer index.


      Returns:
        tensor output from the current layer.
        shape: tf.Tensor(batch_size, new_time_steps, _, filters), dtype=tf.float32)
      """
    cnn_out= tf.compat.v1.layers.conv2d(inputs=inputs,
                                        filters=filters,
                                        kernel_size=kernel_size,
                                        strides=strides,
                                        padding="valid",
                                        activation=tf.nn.relu6,
                                        name="cnn_{}".format(layer_id))
    if is_dropout:
        cnn_out=tf.nn.dropout(cnn_out, rate=0.5)

    return cnn_out

def batch_norm(inputs, training):
    """Batch normalization layer.

    Args:
        inputs: input data for batch norm layer.
        training: a boolean to indicate which stage we are in (training/eval).

   Returns:
    tensor output (exactly the same shape with input)from batch norm layer.

    """

    bnorm_out= tf.compat.v1.layers.batch_normalization(inputs,
                                                       momentum=0.99,
                                                       epsilon=0.001,
                                                       training=training)


    return bnorm_out


def rnn_layer(inputs, rnn_cell, hidden_size, training, layer_id, is_batch_norm=False,is_bidirectional=True, is_dropout=True):
    """Defines RNN layer.

    Args:
        inputs: input tensors for the current layer.
        rnn_cell: RNN cell instance to use.
        rnn_hidden_size: an integer for the dimensionality of the rnn output space.
        training: a boolean to indicate which stage we are in (training/eval).
        layer_id: an integer for the index of current layer.
        is_batch_norm: a boolean specifying whether to perform batch normalization
          on input states.
        is_bidirectional: a boolean specifying whether the rnn layer is
          bi-directional.


    Returns:
        tensor output for the current layer.
        shape: (batch_size, new_time_steps, rnn_hidden_size)
    """

    if is_batch_norm:
        inputs=batch_norm(inputs, training)

    if is_dropout:
        inputs=tf.nn.dropout(inputs, rate=0.5)

    # construct forward/backward RNN cells.
    fw_cell = tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=hidden_size)
    bw_cell = tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=hidden_size)

    if is_bidirectional:
        outputs, _ = tf.compat.v1.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                                               cell_bw=bw_cell,
                                                               inputs=inputs,
                                                               dtype=tf.float32,
                                                               swap_memory=True)
        rnn_outputs = tf.concat(outputs, -1)
    else:
        rnn_outputs = tf.nn.dynamic_rnn(fw_cell, inputs, dtype=tf.float32, swap_memory=True)
    return rnn_outputs

class Model(tf.keras.Model):
    """Defines OCR model."""

    def __init__(self,
                 num_cnn_layers,
                 num_rnn_layers,
                 rnn_type,
                 rnn_hidden_size,
                 num_classes,
                 is_dropout,
                 name="ocr_model",
                 is_bidirectional=True,
                 use_bias=True):

        """Initialize OCR model.

        Args:
            num_cnn_layers: an integer, the number of cnn layers. By default, it's 5
            num_rnn_layers: an integer, the number of rnn layers. By default, it's 5.
            rnn_type: a string, one of the supported rnn cells: gru, rnn and lstm.
            rnn_hidden_size: an integer for the number of hidden states in each unit.
            num_classes: an integer, the number of output classes/labels.
            is_bidirectional: a boolean to indicate if the rnn layer is bidirectional.
            use_bias: a boolean specifying whether to use bias in the last fc layer.
        """

        super(Model, self).__init__(name=name)

        self.num_cnn_layers=num_cnn_layers
        self.num_rnn_layers=num_rnn_layers
        self.rnn_type=rnn_type
        self.is_bidirectional=is_bidirectional
        self.rnn_hidden_size = rnn_hidden_size
        self.num_classes=num_classes,
        self.is_dropout=is_dropout
        self.use_bias=use_bias
    def __call__(self, batch_input, training):
        """
        Make estimations(logits) per time step per class given a batch of input_data.

        Args:
            batch_input: data for the first convolution layer
            training: a boolean to indicate which stage we are in (training/eval).
        Output:
            logits: tensor ouput for loss function (CTC)
            shape (batch_size, new_time_step, n_classes)
        """
        # conv layer
        for cnn_layer_counter in range(self.num_cnn_layers):
            #print(batch_input.shape)
            conv_out = cnn_layer(batch_input, filters=CONV_FILTERS, kernel_size=(33,22),
                                              strides=(2,2), is_dropout=self.is_dropout, layer_id=cnn_layer_counter)

            bNorm_out = batch_norm(conv_out, training)

            #maxPoolOut  = tf.nn.max_pool2d(bNorm_out, ksize=[2,2], strides=[2,2], padding="VALID")


        print("pure output from cnn", bNorm_out.shape)
        # Reshape the output from convolution

        # input has shape (batch_size, H_new, W_new, max_time_step)

        #batch_size = maxPoolOut.shape[0]
        #time_steps=maxPoolOut.get_shape().as_list()[2]
        batch_size = bNorm_out.shape[0]
        time_steps   =  bNorm_out.get_shape().as_list()[3]

        inputs = tf.reshape(bNorm_out,
                            [batch_size, time_steps, -1]) #CONV_FILTERS* time_steps])
        #inputs = tf.reshape(maxPoolOut,
        #                                 [batch_size, -1, CONV_FILTERS* time_steps])

        print("reshaped output from cnn", inputs.shape)

        # RNN Layers
        rnn_cell = RNN_FLAVORS[self.rnn_type]

        for rnn_layer_counter in range(self.num_rnn_layers):

            is_batch_norm= (rnn_layer_counter !=0)

            rnn_out =rnn_layer(inputs,
                               hidden_size=self.rnn_hidden_size,
                               rnn_cell=rnn_cell,
                               is_batch_norm=is_batch_norm,
                               training=training,
                               is_dropout=self.is_dropout,
                               is_bidirectional=self.is_bidirectional,
                               layer_id=rnn_layer_counter)

            #print("rnn_out_shape", rnn_out.shape)

        #  Dense Layer with batch norm
        #b_norm_out2 = batch_norm(rnn_out, training)

        print("final rnn output", rnn_out.shape)
        # fc_layer
        logits = tf.compat.v1.layers.dense(rnn_out, 
                                           106, 
                                           use_bias=self.use_bias, 
                                           activation=tf.nn.relu)
        
        print("shape of logits", logits.shape)
        return logits