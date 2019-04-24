import cv2
import numpy as np
import tensorflow as tf
import os
restore_path = '/home/zhanglu/GBS/tensorflow/NEW_Model/Model_bs_dilated3/model.ckpt-1'
regex_list = ['gru_cell/']
multiple = 10.
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def load_training_list():


    with open('train_list.txt') as f:
        lines = f.read().splitlines()

    files1 = []
    files2 = []
    labels = []
    sals = []
    for line in lines:
        # labels.append('/home/zhanglu/Documents/dataset/DUTS-TR/DUTS-TR-Mask01-extend/%s' % line.replace('.jpg', '.png'))
        # files.append('/home/zhanglu/Documents/dataset/DUTS-TR/DUTS-TR-Image-extend/%s' % line)
        labels.append('/home/zhanglu/Mask_RCNN/train/gt01/%s' % line.replace('.jpg', '.png'))
        files1.append('/home/zhanglu/Mask_RCNN_new/logs/saliency20180610T2239/result1/%s' % line)
        files2.append('/home/zhanglu/Mask_RCNN_new/logs/saliency20180610T2239/result1_pixel/%s' % line)
        # sals.append('/home/zhanglu/Documents/dataset/DUTS-TR/contour-extend/%s' % line.replace('.jpg','.png'))
    return files1, files2, labels, lines


def load_train_val_list():

    files = []
    labels = []

    with open('train_label_list3.txt') as f:
        lines = f.read().splitlines()

    for line in lines:
        labels.append('dataset/MSRA-B/annotation/%s' % line)
        files.append('dataset/MSRA-B/image/%s' % line.replace('.png', '.jpg'))

    with open('dataset/MSRA-B/valid_cvpr2013.txt') as f:
        lines = f.read().splitlines()

    for line in lines:
        labels.append('dataset/MSRA-B/annotation/%s' % line)
        files.append('dataset/MSRA-B/image/%s' % line.replace('.png', '.jpg'))

    return files, labels

def Conv_2d(input_, shape, stddev, name, padding='SAME'):
        with tf.variable_scope(name) as scope:
            W = tf.get_variable('W',
                                shape=shape,
                                initializer=tf.truncated_normal_initializer(stddev=stddev))

            conv = tf.nn.conv2d(input_, W, [1, 1, 1, 1], padding=padding)

            # b = tf.Variable(tf.constant(0.0, shape=[shape[3]]), name='b')
            b = tf.get_variable('b', shape=[shape[3]], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, b)

            return conv

if __name__ == "__main__":


    # model.build_model()
    opt = tf.train.AdamOptimizer(learning_rate=1e-4)
    with tf.variable_scope(tf.get_variable_scope()):

        input1 = tf.placeholder(np.float32,[1,512,512,1],'sal1')
        input2 = tf.placeholder(np.float32, [1, 512, 512, 1], 'sal2')
        label_holder = tf.placeholder(np.float32,[1,512,512,1],'label')
        # input1 = tf.log(input1 / (1.0 - input1))
        # input2 = tf.log(input2 / (1.0 - input2))
        x = tf.concat([input1,input2],3)
        x = Conv_2d(x,[1,1,2,1],0.01,name='combination')
        output = tf.reshape(x,[-1,1])

        label = tf.reshape(label_holder,[-1,1])
        _epsilon = tf.convert_to_tensor(1e-6, output.dtype.base_dtype)

        output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
        Loss_Mean =  tf.reduce_mean(- tf.reduce_sum(label * tf.log(output),
                               len(output.get_shape()) - 1))
        # Loss_Mean = tf.reduce_mean(tf.losses.absolute_difference(labels=label,predictions=x))

        # Loss_Mean = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=label))
        # output = tf.nn.sigmoid(x) >0.5
        # Loss_Mean= tf.reduce_mean(-label* tf.log(x)-(1-label)*tf.log(1-x))
        # # Loss_Mean = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true=label,y_pred=x))
        # output = x
        # model = NM.Model()
        # model.bs_and_dilation()
        max_grad_norm = 1
        tvars = tf.trainable_variables()
        grads = tf.gradients(Loss_Mean, tvars)
        # grads_and_vars = variables_helper.multiply_gradients_matching_regex(zip(grads, tvars), regex_list, multiple)
        # mul_grad = [pair[0] for pair in grads_and_vars]
        # mul_vars = [pair[1] for pair in grads_and_vars]
        # clip_grads, _ = tf.clip_by_global_norm(grads, max_grad_norm)

    train_op = opt.apply_gradients(zip(grads, tvars))
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=100)
    #
    # variables_to_restore = variables_helper.get_variables_available_in_checkpoint(tvars, restore_path)
    # restorer = tf.train.Saver(variables_to_restore)
    # restorer.restore(sess, os.path.join(restore_path))
    # ckpt = tf.train.get_checkpoint_state('val/')
    # saver.restore(sess, ckpt.model_checkpoint_path)
    # # # train_list, label_list = load_train_val_list()
    train1_list, train2_list, label_list, lines= load_training_list()
    n_epochs = 20
    img_size = 512
    label_size = 512
    if not os.path.isdir('val'):
        os.mkdir('val')

    for i in range(1,n_epochs):
        whole_loss = 0.0
        whole_acc = 0.0
        count = 0

        for f_img1, f_img2, f_label, line in zip(train1_list, train2_list, label_list, lines):

            img1 = cv2.imread(f_img1)[:, :, 0].astype(np.float32)
            img_shape = img1.shape
            img1 = cv2.resize(img1, (img_size, img_size))
            img1 = img1.reshape((1, img_size, img_size, 1))
            img1 = img1 / 255.


            img2 = cv2.imread(f_img2)[:, :, 0].astype(np.float32)
            img2 = cv2.resize(img2, (img_size, img_size))
            img2 = img2.reshape((1, img_size, img_size, 1))
            img2 = img2 /255.

            label = cv2.imread(f_label)[:, :, 0].astype(np.float32)
            label = cv2.resize(label, (label_size, label_size))
            label = label.reshape((1,512,512,1))

            # label_c = cv2.imread(sals)[:, :, 0].astype(np.float32)
            # label_c = cv2.resize(label_c, (label_size, label_size))
            # label_c = np.reshape(label_c, [-1, 1])

            _, loss, out = sess.run([train_op,Loss_Mean,output],
                                    feed_dict={input1: img1,
                                               input2: img2,
                                               label_holder: label
                                               })

            whole_loss += loss

            count = count + 1



            if count % 200 == 0:
                out = out.astype(np.float32)
                out = np.reshape(out, [img_size, img_size])
                out = cv2.resize(out, (img_shape[1], img_shape[0]))
                cv2.imwrite('combine/' + line, out * 255)

                print "Loss of %d images: %f" % (count, (whole_loss/count))




        print "Epoch %d: %f" % (i, (whole_loss/len(train1_list)))

        # os.mkdir('Model2')
        saver.save(sess, 'val/model.ckpt', global_step=i)