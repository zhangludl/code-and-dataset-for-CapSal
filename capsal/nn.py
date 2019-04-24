import tensorflow as tf
import tensorflow.contrib.layers as layers

class NN(object):
    def __init__(self):
        self.config = Config
        self.is_train =  False
        self.train_cnn = False
        self.prepare()

    def prepare(self):
        """ Setup the weight initalizers and regularizers. """
        config = self.config

        self.conv_kernel_initializer = layers.xavier_initializer()

        if self.train_cnn and config.conv_kernel_regularizer_scale > 0:
            self.conv_kernel_regularizer = layers.l2_regularizer(
                scale = config.conv_kernel_regularizer_scale)
        else:
            self.conv_kernel_regularizer = None

        if self.train_cnn and config.conv_activity_regularizer_scale > 0:
            self.conv_activity_regularizer = layers.l1_regularizer(
                scale = config.conv_activity_regularizer_scale)
        else:
            self.conv_activity_regularizer = None

        self.fc_kernel_initializer = tf.random_uniform_initializer(
            minval = -config.fc_kernel_initializer_scale,
            maxval = config.fc_kernel_initializer_scale)

        if self.is_train and config.fc_kernel_regularizer_scale > 0:
            self.fc_kernel_regularizer = layers.l2_regularizer(
                scale = config.fc_kernel_regularizer_scale)
        else:
            self.fc_kernel_regularizer = None

        if self.is_train and config.fc_activity_regularizer_scale > 0:
            self.fc_activity_regularizer = layers.l1_regularizer(
                scale = config.fc_activity_regularizer_scale)
        else:
            self.fc_activity_regularizer = None

    def conv2d(self,
               inputs,
               filters,
               kernel_size = (3, 3),
               strides = (1, 1),
               activation = tf.nn.relu,
               use_bias = True,
               name = None):
        """ 2D Convolution layer. """
        if activation is not None:
            activity_regularizer = self.conv_activity_regularizer
        else:
            activity_regularizer = None
        return tf.layers.conv2d(
            inputs = inputs,
            filters = filters,
            kernel_size = kernel_size,
            strides = strides,
            padding='same',
            activation = activation,
            use_bias = use_bias,
            trainable = self.train_cnn,
            kernel_initializer = self.conv_kernel_initializer,
            kernel_regularizer = self.conv_kernel_regularizer,
            activity_regularizer = activity_regularizer,
            name = name)

    def max_pool2d(self,
                   inputs,
                   pool_size = (2, 2),
                   strides = (2, 2),
                   name = None):
        """ 2D Max Pooling layer. """
        return tf.layers.max_pooling2d(
            inputs = inputs,
            pool_size = pool_size,
            strides = strides,
            padding='same',
            name = name)

    def dense(self,
              inputs,
              units,
              activation = tf.tanh,
              use_bias = True,
              name = None):
        """ Fully-connected layer. """
        if activation is not None:
            activity_regularizer = self.fc_activity_regularizer
        else:
            activity_regularizer = None
        return tf.layers.dense(
            inputs = inputs,
            units = units,
            activation = activation,
            use_bias = use_bias,
            trainable = self.is_train,
            kernel_initializer = self.fc_kernel_initializer,
            kernel_regularizer = self.fc_kernel_regularizer,
            activity_regularizer = activity_regularizer,
            name = name)

    def dropout(self,
                inputs,
                name = None):
        """ Dropout layer. """
        return tf.layers.dropout(
            inputs = inputs,
            rate = self.config.fc_drop_rate,
            training = self.is_train)

    def batch_norm(self,
                   inputs,
                   name = None):
        """ Batch normalization layer. """
        return tf.layers.batch_normalization(
            inputs = inputs,
            training = self.train_cnn,
            trainable = self.train_cnn,
            name = name
        )
class Config(object):
    """ Wrapper class for various (hyper)parameters. """
    def __init__(self):
        # about the model architecture
        self.cnn = 'vgg16'               # 'vgg16' or 'resnet50'
        self.max_caption_length = 15
        self.dim_embedding = 512
        self.num_lstm_units = 512
        self.num_initalize_layers = 2    # 1 or 2
        self.dim_initalize_layer = 512
        self.num_attend_layers = 2       # 1 or 2
        self.dim_attend_layer = 512
        self.num_decode_layers = 2       # 1 or 2
        self.dim_decode_layer = 1024

        # about the weight initialization and regularization
        self.fc_kernel_initializer_scale = 0.08
        self.fc_kernel_regularizer_scale = 1e-4
        self.fc_activity_regularizer_scale = 0.0
        self.conv_kernel_regularizer_scale = 1e-4
        self.conv_activity_regularizer_scale = 0.0
        self.fc_drop_rate = 0.5
        self.lstm_drop_rate = 0.3
        self.attention_loss_factor = 0.01

        # about the optimization
        self.num_epochs = 100
        self.batch_size = 1#3264
        self.optimizer = 'Adam'    # 'Adam', 'RMSProp', 'Momentum' or 'SGD'
        self.initial_learning_rate = 0.0001#0.0001
        self.learning_rate_decay_factor = 1.0
        self.num_steps_per_decay = 100000
        self.clip_gradients = 5.0
        self.momentum = 0.0
        self.use_nesterov = True
        self.decay = 0.9
        self.centered = True
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-6

        # about the saver
        self.save_period = 1000
        self.save_dir = './models/'
        self.summary_dir = './summary/'

        # about the vocabulary
        self.vocabulary_file = './vocabulary.csv'
        self.vocabulary_size = 5000

        # about the training
        self.train_feature_dir = '/home/zhanglu/Mask_RCNN_new/logs/feat/'
        self.train_image_dir = '/home/zhanglu/Documents/Referring Image Segmentation/MSCOCO/2014/train2014/'
        self.train_caption_file = '/home/zhanglu/Documents/Referring Image Segmentation/MSCOCO/2014/annotations/captions_train2014.json'
        self.temp_annotation_file = './train/anns.csv'
        self.temp_data_file = './train/data.npy'

        # about the evaluation
        self.eval_feature_dir = '/home/zhanglu/Mask_RCNN_new/val_feat/'
        self.eval_image_dir = '/home/zhanglu/Documents/Referring Image Segmentation/MSCOCO/2014/val2014/'
        self.eval_caption_file = '/home/zhanglu/Documents/Referring Image Segmentation/MSCOCO/2014/annotations/captions_val2014.json'
        self.eval_result_dir = './val/results/'
        self.eval_result_file = './val/results.json'
        self.save_eval_result_as_image = False
        self.eval_my_image_dir = '/home/zhanglu/Mask_RCNN/val/val256/'#/home/zhanglu/Mask_RCNN/train/train256/
        self.eval_my_save_dir = '/home/zhanglu/image_captioning/feat/train/'
        # about the testing
        self.test_image_dir = './test/images/'
        self.test_result_dir = './test/results/'
        self.test_result_file = './test/results.csv'
