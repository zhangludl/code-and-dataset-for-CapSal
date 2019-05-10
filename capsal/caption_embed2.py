import tensorflow as tf
import numpy as np
import tensorflow.contrib.layers as layers
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
from keras.initializers import Constant

def fc_kernel_initializer():
    return tf.random_uniform_initializer(minval = -0.08,maxval = 0.08)
def fc_kernel_regularizer():
    return layers.l2_regularizer(scale = 1e-4)
def load_weight_caption():
    model_dir = './model/keras_caption.npy'
    data_dict = np.load(model_dir,encoding="bytes").item()
    count = 0
    sess = tf.Session()
    K.set_session(sess)
    sess.run(tf.global_variables_initializer())
    for v in tqdm(tf.global_variables()):
        if v.name in data_dict.keys():
            sess.run(v.assign(data_dict[v.name]))
            count += 1
    print("%d tensors loaded." % count)
    print("image caption load")
def load_weight():
    model_dir = '/home/zhanglu/Mask_RCNN_new/mrcnn/keras_caption.npy'
    data_dict = np.load(model_dir,encoding='latin1').item()
    return data_dict


def build_rnn(input,config):
    ctx = 64
    down = KL.Conv2D(512, (3, 3), padding="same", activation="relu", name='gcap_down_imagefeature')(input)

    reshaped_conv5_3_feats = KL.Lambda(lambda x: tf.reshape(x, [config.BATCH_SIZE, ctx, 512]))(down)
    conv_feats = reshaped_conv5_3_feats

    print("Building the RNN...")

    contexts = conv_feats
    reshaped_contexts = KL.Lambda(lambda x: tf.reshape(x, [-1, 512]))(contexts)
    temp1 = attend_1(reshaped_contexts)
    w_embedding = KL.Embedding(input_dim=5000, output_dim=512, name='gcap_embedding')

    # Setup the LSTM

    # Initialize the LSTM using the mean context
    # with tf.variable_scope("initialize"):
    context_mean = KL.Lambda(lambda x: tf.reduce_mean(x, axis=1))(conv_feats)
    initial_memory, initial_output = initialize(context_mean)
    initial_state = initial_memory, initial_output

    # Prepare to run
    predictions = []
    outputs = []
    current_inputs = []
    num_steps = 15
    last_output = initial_output
    last_memory = initial_memory
    last_word = KL.Lambda(lambda x: K.zeros([config.BATCH_SIZE], 'int32'))(input)
    last_state = last_output, last_memory
    alphas = []
    att_masks = []
    cross_entropies = []
    predictions_correct = []
    lstm = KL.LSTM(512, return_state=True, recurrent_activation='hard_sigmoid', name='gcap_lstm',
                   unit_forget_bias=False)  # (last_output,initial_state = initial_state)

    # Generate the words one by one
    for idx in range(num_steps):
        # Attention mechanism
        # with tf.variable_scope("attend"):
        # alpha = attend(reshaped_contexts, last_output)

        # use 2 fc layers to attend

        temp2 = attend_2(last_output)

        temp2 = KL.Lambda(lambda x: tf.reshape(tf.tile(tf.expand_dims(x, 1), [1, ctx, 1]), [-1, 512]))(temp2)
        temp = KL.Add()([temp1,temp2])
        att_logits = attend_3(temp)
        att_logits = KL.Lambda(lambda x: tf.reshape(x, [-1, ctx]))(att_logits)
        alpha = KL.Softmax()(att_logits)
        alpha1 = KL.RepeatVector(512)(alpha)
        alpha1 = KL.Permute((2, 1))(alpha1)
        context = KL.Multiply()([contexts, alpha1])
        context = KL.Lambda(lambda x: tf.reduce_sum(x,
                                                    axis=1))(context)
        alphas.append(alpha)
        word_embed = w_embedding(last_word)
        # Apply the LSTM
        # with tf.variable_scope("lstm"):

        current_input = KL.Concatenate(axis=-1)([context, word_embed])
        current_input = KL.Lambda(lambda x: tf.expand_dims(x, 1))(current_input)

        output, memory, cell_out = lstm(current_input, initial_state=list(last_state))  #
        state = memory, cell_out
        current_inputs.append(current_input)
        outputs.append(output)
        # Decode the expanded output of LSTM into a word
        # with tf.variable_scope("decode"):

        expanded_output = KL.Concatenate(axis=-1)([output,
                                                   context,
                                                   word_embed])
        logits = decode(expanded_output)
        # probs = KL.Lambda(lambda x: tf.nn.softmax(logits))(logits)
        prediction = KL.Lambda(lambda x: tf.argmax(x, 1))(logits)
        predictions.append(prediction)




        last_output = output
        last_memory = memory
        last_state = state
        if idx == 0:
            att_mask = KL.Lambda(lambda x: K.switch(tf.equal(x[0], 0), tf.constant(0.0), tf.constant(1.0)))(last_word)
        else:
            att_mask = KL.Lambda(lambda x: K.switch(tf.equal(x[0], 2), tf.constant(0.0), tf.constant(1.0)))(last_word)
        att_masks.append(att_mask)
        last_word = KL.Lambda(lambda x: tf.cast(x, tf.int32))(prediction) #


        # tf.get_variable_scope().reuse_variables()

        # Compute the final loss, if necessary

    outputs = KL.Lambda(lambda x: tf.reshape(x, [config.BATCH_SIZE, num_steps, 512]))(outputs)
    predictions = KL.Lambda(lambda x: tf.reshape(tf.cast(x, tf.float32), [config.BATCH_SIZE, num_steps, 1]))(
        predictions)
    att_masks = KL.Lambda(lambda x: tf.reshape(tf.cast(x, tf.float32), [num_steps, 1,1,1]))(
        att_masks)
    alphas = KL.Lambda(lambda x: tf.reshape(x,[config.BATCH_SIZE, num_steps, ctx]))(alphas)

    print("RNN built.")
    return outputs, predictions,alphas,att_masks
def build_rnn2(input,caption_gt,masks,config):

    down = KL.Conv2D(512, (3, 3), padding="same", activation="relu", name='gcap_down_imagefeature')(input)

    reshaped_conv5_3_feats = KL.Lambda(lambda x:tf.reshape(x, [config.BATCH_SIZE, 64, 512]))(down)
    conv_feats = reshaped_conv5_3_feats

    print("Building the RNN...")

    contexts = conv_feats
    reshaped_contexts = KL.Lambda(lambda x: tf.reshape(x, [-1, 512]))(contexts)
    temp1 = attend_1(reshaped_contexts)
    w_embedding = KL.Embedding(input_dim=5000,output_dim=512,name='gcap_embedding')


        # Setup the LSTM

    # Initialize the LSTM using the mean context
    # with tf.variable_scope("initialize"):
    context_mean = KL.Lambda(lambda x: tf.reduce_mean(x, axis=1))(conv_feats)
    initial_memory, initial_output = initialize(context_mean)
    initial_state = initial_memory, initial_output

        # Prepare to run
    predictions = []
    outputs = []
    current_inputs = []
    num_steps = 15
    last_output = initial_output
    last_memory = initial_memory
    last_word = KL.Lambda(lambda x: K.zeros([config.BATCH_SIZE],'int32'))(input)
    last_state = last_output,last_memory
    alphas = []
    cross_entropies = []
    predictions_correct = []
    lstm = KL.LSTM(512,return_state=True,recurrent_activation='hard_sigmoid',name='gcap_lstm',
                   unit_forget_bias=False)#(last_output,initial_state = initial_state)

    # Generate the words one by one
    for idx in range(num_steps):
        # Attention mechanism
        # with tf.variable_scope("attend"):
        # alpha = attend(contexts, last_output)

        # use 2 fc layers to attend

        temp2 = attend_2(last_output)

        temp2 = KL.Lambda(lambda x: tf.reshape(tf.tile(tf.expand_dims(x, 1), [1, 64, 1]), [-1, 512]))(temp2)
        temp = KL.Add()([temp1,temp2])
        att_logits = attend_3(temp)
        att_logits = KL.Lambda(lambda x: tf.reshape(x, [-1, 64]))(att_logits)
        alpha = KL.Softmax()(att_logits)
        alpha1 = KL.RepeatVector(512)(alpha)
        alpha1 = KL.Permute((2,1))(alpha1)
        context = KL.Multiply()([contexts,alpha1])
        context = KL.Lambda(lambda x: tf.reduce_sum(x,
                                    axis=1))(context)
        tiled_masks = KL.Lambda(lambda x: tf.tile(tf.expand_dims(x[:, idx], 1),[1, 64]))(masks)
        masked_alpha = KL.Lambda(lambda x: tf.reshape(x * tiled_masks,[-1]))(alpha)
        alphas.append(masked_alpha)

        word_embed = w_embedding(last_word)
            # Apply the LSTM
        # with tf.variable_scope("lstm"):

        current_input = KL.Concatenate(axis=-1)([context, word_embed])
        current_input = KL.Lambda(lambda x: tf.expand_dims(x,1))(current_input)


        output, memory, cell_out = lstm(current_input, initial_state = list(last_state))#
        state = memory, cell_out
        current_inputs.append(current_input)
        outputs.append(output)
            # Decode the expanded output of LSTM into a word
        # with tf.variable_scope("decode"):

        expanded_output = KL.Concatenate(axis = -1)([output,
                                         context,
                                         word_embed])
        logits = decode(expanded_output)
        # probs = KL.Lambda(lambda x: tf.nn.softmax(logits))(logits)
        prediction = KL.Lambda(lambda x: tf.argmax(x, 1))(logits)
        predictions.append(prediction)

        # Compute the loss for this step, if necessary
        masked_cross_entropy = KL.Lambda(lambda x: caption_loss(*x))([caption_gt[:,idx],logits,masks[:,idx]])
        cross_entropies.append(masked_cross_entropy)

        # ground_truth = KL.Lambda(lambda x: tf.cast(caption_gt[:, idx], tf.int64))(caption_gt)
        # prediction_correct = tf.where(
        #     tf.equal(prediction, ground_truth),
        #     tf.cast(masks[:, idx], tf.float32),
        #     tf.cast(tf.zeros_like(prediction), tf.float32))
        # predictions_correct.append(prediction_correct)

        last_output = output
        last_memory = memory
        last_state = state
        last_word = KL.Lambda(lambda x: tf.reshape(tf.cast(x[:,idx], tf.int32),[config.BATCH_SIZE]))(caption_gt)  #

        # tf.get_variable_scope().reuse_variables()

        # Compute the final loss, if necessary
    cross_entropies = KL.Lambda(lambda x : tf.stack(x, axis=1))(cross_entropies)
    cross_entropy_loss = KL.Lambda(lambda x: tf.reduce_sum(x) / tf.reduce_sum(masks))(cross_entropies)

    alphas = KL.Lambda(lambda x: tf.reshape(tf.stack(x, axis=1),[1,64,-1]))(alphas)
    attentions = KL.Lambda(lambda x: tf.reduce_sum(x, axis=2))(alphas)
    diffs = KL.Lambda(lambda x: tf.ones_like(x) - x)(attentions)
    attention_loss = KL.Lambda(lambda x: 0.01 * tf.nn.l2_loss(x) / (64))(diffs)


    total_loss = KL.Lambda(lambda x:cross_entropy_loss + x,name="caption_loss")(attention_loss)

    outputs = KL.Lambda(lambda x: tf.reshape(x,[config.BATCH_SIZE,num_steps,512]))(outputs)
    predictions = KL.Lambda(lambda x: tf.reshape(tf.cast(x,tf.float32),[config.BATCH_SIZE,num_steps,1]))(predictions)
    # outputs2 = KL.Lambda(lambda x: tf.concat([outputs,predictions],axis=0))(outputs)

    print("RNN built.")
    return outputs, predictions, total_loss
def caption_loss(label,prediction,mask):

    cross_entropy =K.sparse_categorical_crossentropy(target=label,output=prediction,from_logits=True)
    masked_cross_entropy = mask * cross_entropy
    return masked_cross_entropy

def initialize( context_mean):
    # use 2 fc layers to initialize
    temp1 = KL.Dense(512,activation='tanh',name='gcap_initialize_fc_a1')(context_mean)#
    memory = KL.Dense(512, name='gcap_initialize_fc_a2')(temp1)#
    temp2 = KL.Dense(512, activation='tanh', name='gcap_initialize_fc_b1')(context_mean)#

    output = KL.Dense(512, name='gcap_initialize_fc_b2')(temp2)
    return memory, output
attend_1 = KL.Dense(512,activation='tanh',name='gcap_attend_fc_1a')#
attend_2 = KL.Dense(512,activation='tanh',name='gcap_attend_fc_1b')#
attend_3 = KL.Dense(1,use_bias=False,name='gcap_attend_fc_2')#
def attend(inpu, output):

    # """ Attention Mechanism. """

    # reshaped_contexts = KL.Lambda(lambda x: tf.reshape(x, [-1, 512]))(contexts)
    # use 2 fc layers to attend
    temp1 = attend_1(inpu)
    temp2 = attend_2(output)

    temp2 = KL.Lambda(lambda x: tf.tile(tf.expand_dims(temp2, 1), [1, 64, 1]))(temp2)
    temp2 = KL.Lambda(lambda x: tf.reshape(temp2, [-1, 512]))(temp2)
    temp = KL.Lambda(lambda x: temp1 + x)(temp2)
    logits = attend_3(temp)
    logits = KL.Lambda(lambda x: tf.reshape(logits, [-1, 64]))(logits)
    alpha = KL.Lambda(lambda x: tf.nn.softmax(logits))(logits)
    return alpha
decode_1 = KL.Dense(1024,activation='tanh',name='gcap_decode_fc_1')#
decode_2 = KL.Dense(5000,activation=None,name='gcap_decode_fc_2')#
def decode(expanded_output):
    # """ Decode the expanded output of the LSTM into a word. """
    # use 2 fc layers to decode
    temp = decode_1(expanded_output)

    logits = decode_2(temp)

    return logits
# def initialize( context_mean):
#     # use 2 fc layers to initialize
#     temp1 = KL.Dense(512,activation='tanh',kernel_initializer=fc_kernel_initializer(),name='fc_a1')(context_mean)
#     memory = KL.Dense(512, kernel_initializer=fc_kernel_initializer(), name='fc_a2')(temp1)
#     temp2 = KL.Dense(512, activation='tanh', kernel_initializer=fc_kernel_initializer(), name='fc_b1')(context_mean)
#     output = KL.Dense(512, kernel_initializer=fc_kernel_initializer(), name='fc_b2')(temp2)
#     return memory, output
#
# def attend(contexts, output):
#
#     # """ Attention Mechanism. """
#
#     reshaped_contexts = tf.reshape(contexts, [-1, 512])
#     # use 2 fc layers to attend
#     temp1 = KL.Dense(512, activation='tanh', kernel_initializer=fc_kernel_initializer(),name='fc_1a')(reshaped_contexts)
#     temp2 = KL.Dense(512, activation='tanh', kernel_initializer=fc_kernel_initializer(), name='fc_1b')(
#         output)
#     temp2 = tf.tile(tf.expand_dims(temp2, 1), [1, 64, 1])
#     temp2 = tf.reshape(temp2, [-1, 512])
#     temp = temp1 + temp2
#     logits = KL.Dense(1, use_bias=False,kernel_initializer=fc_kernel_initializer(), name='fc_2')(
#         temp)
#     logits = tf.reshape(logits, [-1, 64])
#     alpha = tf.nn.softmax(logits)
#     return alpha
#
#
# def decode(expanded_output):
#     # """ Decode the expanded output of the LSTM into a word. """
#     # use 2 fc layers to decode
#     temp = KL.Dense(1024, activation='tanh', kernel_initializer=fc_kernel_initializer(), name='fc_1')(
#         expanded_output)
#     logits = KL.Dense(5000,  kernel_initializer=fc_kernel_initializer(), name='fc_2')(
#         temp)
#     return logits
def conv2d(input_,shape,activation = tf.nn.relu,padding = 'SAME',name = None):
    with tf.variable_scope(name) as scope:
        W = tf.get_variable('kernel',
                            shape=shape,
                            initializer=tf.truncated_normal_initializer(stddev=0.01))

        conv = tf.nn.conv2d(input_, W, [1, 1, 1, 1], padding=padding)

        # b = tf.Variable(tf.constant(0.0, shape=[shape[3]]), name='b')
        b = tf.get_variable('bias', shape=[shape[3]], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, b)
        conv = activation(conv)
    return conv
def dense(input,shape,use_bias = True,name = None):

    with tf.variable_scope(name) as scope:
        weight = tf.get_variable('kernel',shape=shape,initializer=fc_kernel_initializer())
        if use_bias:
            bias = tf.get_variable('bias',shape = [shape[1]],initializer=tf.constant_initializer(0.0))
            out = tf.nn.bias_add(tf.matmul(input, weight), bias)
        else:
            out = tf.matmul(input, weight)

        return out
