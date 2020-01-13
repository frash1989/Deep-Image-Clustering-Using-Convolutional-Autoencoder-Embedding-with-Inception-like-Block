from time import time
import numpy as np
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
from mywork.libs.utils import corrupt
import math
from math import log
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import normalized_mutual_info_score
import os
from datetime import datetime
import time
import functools
from functools import partial
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
from sklearn import preprocessing
from dataset_batch_process import *
import scipy.io as sio
np.random.seed(seed=0)
SEED =66478
tf.set_random_seed(seed =SEED)
#聚类Accuracy计算
def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def inference(input):
    #output size 14*14*32
    conv1 = tf.layers.conv2d(input, filters=32, kernel_size=(5, 5), strides=(2, 2), padding='same',activation=tf.nn.relu, name='conv1')
    # output size 7*7*64
    conv2 = tf.layers.conv2d(conv1, filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same',activation=tf.nn.relu, name='conv2')
    # output size 3*3*128
    conv3 = tf.layers.conv2d(conv2, filters=128, kernel_size=(3, 3), strides=(2, 2), padding='VALID',activation=tf.nn.relu, name='conv3')

    num_batch, height, width, num_channels = conv3.get_shape()
    #展开tensor，行数未知（其实为batch数目），列数为(height * width * num_channels).value
    unfold = tf.reshape(conv3, [-1, (height * width * num_channels).value])

    encoded = tf.layers.dense(unfold,units=10,activation=None,name='encoded')

    decoded = tf.layers.dense(encoded,units=(height * width * num_channels).value,activation=tf.nn.relu,name='decoded')
    # decoded =tf.layers.batch_normalization(decoded)
    # decoded = tf.nn.relu(decoded,name='decoded')
    fold_shape = [-1, height.value, width.value, num_channels.value]
    # 折叠tensor，行数未知（其实为batch数目），其它三个维度为[height.value  width.value  num_channels.value]
    # output size 3*3*128
    fold = tf.reshape(decoded, fold_shape, name='fold')
    # output size 7*7*64
    deconv1 = tf.layers.conv2d_transpose(fold,
                                         filters=64,
                                         kernel_size=(3, 3),
                                         strides=(2, 2),
                                         padding='VALID',
                                         activation=tf.nn.relu,
                                         name='deconv1')
    # output size 14*14*32
    deconv2 = tf.layers.conv2d_transpose(deconv1,
                                         filters=32,
                                         kernel_size=(5, 5),
                                         strides=(2, 2),
                                         padding='same',
                                         activation=tf.nn.relu,
                                         name='deconv2')
    # output size 28*28*1
    deconv3 = tf.layers.conv2d_transpose(deconv2,
                                         filters=1,
                                         kernel_size=(5, 5),
                                         strides=(2, 2),
                                         padding='same',
                                         activation=None,
                                         name='deconv3')

    reconstruct = deconv3
    parameters = {'input': input,
                  'conv_1': conv1,
                  'conv_2': conv2,
                  'conv_3': conv3,
                  'unfold': unfold,
                  'encoded': encoded,
                  'decoded': decoded,
                  'fold': fold,
                  'deconv_1': deconv1,
                  'deconv_2': deconv2,
                  'deconv_3': deconv3,
                  'reconstruct': reconstruct}
    return parameters
def kl_divergence(p, q):  # 计算KL散度
    return tf.reduce_sum(p * tf.log(p / q))
def confidence_assign(embedding, center):  # 计算q的值
    alpha = 1  # 默认为1
    dist = tf.reduce_sum(tf.square(tf.expand_dims(embedding, 1) - center), 2)
    q = 1.0 / (1.0 + dist / alpha)
    q **= (alpha + 1.0) / 2.0
    q = tf.transpose(tf.transpose(q) / tf.reduce_sum(q, 1))
    return q
def target_distribution(q):  # 计算目标分布，即p值
    # 常用方式
    # weight = q ** 2 / tf.reduce_sum(q, 0)

    #FCAE使用的方式
    tmp_q = tf.square(q)
    weight = q ** 2 / tf.reduce_sum(tmp_q, 0)
    return tf.transpose(tf.transpose(weight) / tf.reduce_sum(weight, 1))

def target_distribution_numpy(q):
    # 常用方式
    weight = q ** 2 / q.sum(0)
    # FCAE使用的方式
    # tmp_q = np.square(q)
    # weight = q ** 2 / tmp_q.sum(0)
    return (weight.T / weight.sum(1)).T

def clustering(sess, net_in,net_out,data,label):
    #sess: session
    #net_in: placeholder
    #net_out: 网络的输出
    #data，label表示输入数据的情况，label是one_hot编码的
    #################################聚类###########################################
    tmp_feat = []
    for ii in range(len(data) // 100):
        batch_xs = data[ii * 100:(ii + 1) * 100, :]
        input_image = np.reshape(batch_xs, [len(batch_xs), 28, 28, 1])
        net_output = sess.run(net_out['encoded'], feed_dict={net_in: input_image})
        tmp_feat = list(tmp_feat) + list(np.squeeze(net_output))

    total_feat = np.array(tmp_feat)
    start_time = time.time()
    kmeans = KMeans(n_clusters=10, n_init=20)  # n_clusters表示聚类的个数，n_init用不同的初始化质心运行算法的次数，默认为10
    y_pred = kmeans.fit_predict(total_feat)
    duration = time.time() - start_time
    print("Runtime of kmeans clustering = %.3f" % duration)
    y = label.argmax(1)  # 获得每行最大元素索引
    ACC = cluster_acc(y, y_pred)
    NMI = normalized_mutual_info_score(y, y_pred)
    print('ACC = %.2f   NMI = %.2f' % (ACC * 100, NMI * 100))

def feat_batch(sess, net_in,net_out,data,batch_size):
    if np.mod(len(data),batch_size) ==0:
        tmp_feat = []
        num_batches = len(data) // batch_size
        for ii in range(num_batches):
            batch_xs = data[ii * batch_size:(ii + 1) * batch_size, :]
            input_image = np.reshape(batch_xs, [len(batch_xs), 28, 28, 1])
            net_output = sess.run(net_out, feed_dict={net_in: input_image})
            tmp_feat = list(tmp_feat) + list(np.squeeze(net_output))

    else:
        tmp_feat = []
        num_batches = (len(data) // batch_size)+1
        for ii in range(num_batches-1):
            batch_xs = data[ii * batch_size:(ii + 1) * batch_size, :]
            input_image = np.reshape(batch_xs, [len(batch_xs), 28, 28, 1])
            net_output = sess.run(net_out, feed_dict={net_in: input_image})
            tmp_feat = list(tmp_feat) + list(np.squeeze(net_output))

        batch_end = data[(num_batches-1) * batch_size ::, :]
        left_num = num_batches*batch_size - len(data)
        batch_xs_left = data[0:left_num, :]
        batch_xs = np.vstack((batch_end,batch_xs_left))
        input_image = np.reshape(batch_xs, [len(batch_xs), 28, 28, 1])
        net_output = sess.run(net_out, feed_dict={net_in: input_image})
        tmp_feat = list(tmp_feat) + list(np.squeeze(net_output))

    total_feat = np.array(tmp_feat)

    return total_feat[0:len(data),:]

def clustering_batch(sess, net_in,net_out,data,label,batch_size):
    #sess: session
    #net_in: placeholder
    #net_out: 网络的输出
    #data，label表示输入数据的情况，label是one_hot编码的
    #################################聚类###########################################
    total_feat = feat_batch(sess, net_in,net_out,data,batch_size)
    start_time = time.time()
    kmeans = KMeans(n_clusters=10, n_init=20)  # n_clusters表示聚类的个数，n_init用不同的初始化质心运行算法的次数，默认为10
    y_pred = kmeans.fit_predict(total_feat)
    duration = time.time() - start_time
    # print("Runtime of kmeans clustering = %.3f" % duration)
    y = label.argmax(1)  # 获得每行最大元素索引
    ACC = cluster_acc(y, y_pred)
    NMI = normalized_mutual_info_score(y, y_pred)
    print('ACC = %.2f   NMI = %.2f' % (ACC * 100, NMI * 100))


def test_mnist():
    # load MNIST as before
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    print("Train samples of mnist %d" % mnist.train.num_examples)
    print("Validation samples of mnist %d" % mnist.validation.num_examples)
    print("Test samples of mnist %d" % mnist.test.num_examples)

    print(type(mnist.train.images))
    print(len(mnist.train.images))  # 行数
    print(mnist.train.images.size)  # 元素个数
    print(mnist.train.images.shape)  # 行数，列数

    mnist_total_images =np.vstack((mnist.train.images,mnist.validation.images,mnist.test.images))
    mnist_total_labels = np.vstack((mnist.train.labels, mnist.validation.labels, mnist.test.labels))
    #################################聚类###########################################
    mnist_input_images = mnist_total_images
    mnist_input_labels = mnist_total_labels

    # mnist_input_images = mnist.test.images
    # mnist_input_labels = mnist.test.labels
    ###################parameter setting##########################
    batch_size = 256
    n_epochs_finetune = 100000

    initial_learning_rate = 1e-3  # 初始学习率
    display_step = 1000
    cluster_display_step =5000
    ##==================================================
    X = tf.placeholder("float", [batch_size, 28,28,1])
    # batch_size_test = 100
    global_step = tf.Variable(0,trainable=False)

    para = inference(X)
    shuffle_input = Dataset(mnist_input_images) #shuffle后数据
    mnist_index = np.arange(0,len(mnist_input_images))
    shuffle_dec_input= DatasetWithIndex(mnist_input_images,mnist_index)  # shuffle后数据
    # learning_rate —— 事先设定的初始学习率，
    # global_step —— 当前迭代轮数
    # decay_steps —— 衰减速度，即多少轮可以迭代完一次所有样本数据
    # decay_rate —— 衰减系数
    # staircase —— True表示成阶梯函数下降，False时表示连续衰减
    # 学习率每隔20000步 则乘以0.1
    # add_global = global_step.assign_add(1)
    learning_rate = tf.train.exponential_decay(learning_rate=initial_learning_rate,
                                               global_step=global_step,
                                               decay_steps=30000,
                                               decay_rate=0.1,
                                               staircase=True)
    # MSE loss
    mse_loss= tf.reduce_mean(tf.square(tf.subtract(X , para['reconstruct'])))
    total_cost = mse_loss
    # total_cost =  mse_loss
    # RMSE loss
    # total_cost = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(X_layer1, para['output']))))
    #Cross Entropy
    #Note: cross-entropy should only be used when the values are between 0 and 1.
    # epsilon =1e-10
    # cross_entropy = -1. * X_layer1 * tf.log(para['output']+epsilon) - (1. - X_layer1) * tf.log(1. - para['output']+epsilon)
    # total_cost = tf.reduce_mean(tf.reduce_sum(cross_entropy,reduction_indices=[1]))
    # total_cost = -tf.reduce_mean(tf.reduce_sum(X_layer1 * tf.log(para['output']), reduction_indices=[1]))

    # optimizer_total = tf.train.AdamOptimizer(learning_rate).minimize(total_cost)
    optimizer_total = tf.train.AdamOptimizer(learning_rate).minimize(total_cost)
    # optimizer_total = tf.train.AdamOptimizer(learning_rate).minimize(total_cost,global_step=global_step)
    # optimizer_total = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(total_cost)
    # optimizer_total = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(total_cost,global_step=global_step)
    # optimizer_total = tf.train.AdadeltaOptimizer(learning_rate).minimize(total_cost)
    # optimizer_total = tf.train.AdadeltaOptimizer(learning_rate).minimize(total_cost, global_step=global_step)
    # optimizer_total = tf.train.GradientDescentOptimizer(learning_rate).minimize(total_cost,global_step=global_step)
    # optimizer_total = tf.train.RMSPropOptimizer(learning_rate).minimize(total_cost,global_step=global_step)
    # optimizer_total = tf.train.RMSPropOptimizer(learning_rate).minimize(total_cost)
    # optimizer_total = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(total_cost, global_step=global_step)
    # optimizer_total = tf.train.MomentumOptimizer(learning_rate,0.9).minimize(total_cost)
    ############################Details of DEC###############################
    dec_epoch=10000000
    dec_display_step=1000
    dec_cluster_display_step=1000
    num_clusters = 10
    update_interval =  5#1*(len(mnist_input_labels)//batch_size +1)#10000#
    tol             = 1e-3
    alpha = 0.001

    centroides = tf.Variable(tf.random_normal([num_clusters, 10], stddev=0.01))
    q = confidence_assign(para['encoded'], centroides)
    p_x = tf.placeholder("float", [None, num_clusters])
    kl_loss = kl_divergence(p_x, q)

    initial_learning_rate_dec = 1e-4

    learning_rate_dec = tf.train.exponential_decay(learning_rate=initial_learning_rate_dec,
                                               global_step=global_step,
                                               decay_steps=10000,
                                               decay_rate=0.9,
                                               staircase=True)
    # total_loss = alpha*kl_loss +(1-alpha)*mse_loss
    total_loss = kl_loss
    optimizer_dec = tf.train.AdamOptimizer(learning_rate_dec).minimize(total_loss)

    all_variables = tf.all_variables()
    restore = ['conv1/kernel:0', 'conv1/bias:0',
               'conv2/kernel:0', 'conv2/bias:0',
               'conv3/kernel:0', 'conv3/bias:0',
                'encoded/kernel:0', 'encoded/bias:0',
                'decoded/kernel:0', 'decoded/bias:0',
               'deconv1/kernel:0', 'deconv1/bias:0',
               'deconv2/kernel:0', 'deconv2/bias:0',
               'deconv3/kernel:0', 'deconv3/bias:0'
               ]


    restore_var = [v for v in tf.all_variables() if v.name in restore]  # Keep only the var
    print(restore_var)
    # help(tf.train.Saver)
    all_saver = tf.train.Saver(restore_var)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    #########################################################################################
    ckpt_dir = '../trained_models/mnist_full/CAE'
    # ckpt_dir = '../trained_models/mnist_test/CAE'
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        all_saver.restore(sess, ckpt.model_checkpoint_path)
        print('Restoring saved parameters!')
        #################################聚类###########################################
        # clustering(sess, X, para, mnist_input_images, mnist_input_labels)
        clustering_batch(sess, X, para['encoded'], mnist_input_images, mnist_input_labels, batch_size)
    #########################################################################################
    else:
        print('Train networks!')
        clustering_batch(sess, X, para['encoded'], mnist_input_images, mnist_input_labels, batch_size)
        start_time = time.time()
        duration = 0
        for epoch_i in range(n_epochs_finetune):
            global_value, rate = sess.run([global_step, learning_rate])
            train = shuffle_input.next_batch(batch_size)
            train_image = np.reshape(train,[batch_size,28,28,1])
            _, c_total,mse_value,= sess.run([optimizer_total, total_cost,mse_loss], feed_dict={X: train_image})
            if epoch_i == 0:
                print('global_value=', global_value, '   ', 'rate = ', rate)
                time_cost = time.time() - start_time - duration
                duration = time.time() - start_time
                print("Epoch:", '%04d' % (epoch_i + 1),  "duration = %.3f" % duration, "time_cost = %.3f" % time_cost,"mse_loss=", "{:.9f}".format(mse_value))
                # print("cost=", "{:.9f}".format(c_total),"mse_loss=", "{:.9f}".format(mse_value),"reg_value=", "{:.9f}".format(reg_value) )
                print('**')
            if (epoch_i+1) % display_step == 0:
                # print('global_value=', global_value, '   ', 'rate = ', rate)
                time_cost = time.time() - start_time - duration
                duration = time.time() - start_time
                print("Epoch:", '%04d' % (epoch_i + 1),  "duration = %.3f" % duration, "time_cost = %.3f" % time_cost,"mse_loss=", "{:.9f}".format(mse_value))
                # print("cost=", "{:.9f}".format(c_total),"mse_loss=", "{:.9f}".format(mse_value),"reg_value=", "{:.9f}".format(reg_value) )

            if (epoch_i + 1) % cluster_display_step == 0:
                # clustering(sess, X, para, mnist_input_images, mnist_input_labels)
                clustering_batch(sess, X, para['encoded'], mnist_input_images, mnist_input_labels,batch_size)

        print("Total Optimization Finished!")
        save_path_all = all_saver.save(sess, "../trained_models/mnist_full/CAE/dec_model.ckpt")
        # save_path_all = all_saver.save(sess, "../trained_models/mnist_test/CAE/dec_model.ckpt")
        print("All Model saved in file: ", save_path_all)

    # clustering(sess, X, para, mnist_input_images, mnist_input_labels)
    clustering_batch(sess, X, para['encoded'], mnist_input_images, mnist_input_labels, batch_size)

    print('Train CAE networks!')
    print('centroids initialization !')

    total_feat = feat_batch(sess, X, para['encoded'], mnist_input_images, batch_size)
    embedding_feature = total_feat
    kmeans = KMeans(n_clusters=10, n_init=20)  # n_clusters表示聚类的个数，n_init用不同的初始化质心运行算法的次数，默认为10
    y_pred = kmeans.fit_predict(embedding_feature)
    y_pred_last = y_pred
    temp_cent2 = sess.run(tf.assign(centroides, kmeans.cluster_centers_))
    y = mnist_input_labels.argmax(1)  # 获得每行最大元素索引

    total_value = 0
    mse_value = 0
    kl_value = 0
    start_time = time.time()
    duration = 0
    for epoch_i in range(dec_epoch):
        if epoch_i % update_interval==0:

            q_value = feat_batch(sess, X, q, mnist_input_images, batch_size)
            p_value = target_distribution_numpy(q_value)
            # evaluate the clustering performance
            y_pred = q_value.argmax(1)
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
            y_pred_last = y_pred
            acc = np.round(cluster_acc(y, y_pred), 5)
            nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
            ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)

            time_cost = time.time() - start_time - duration
            duration = time.time() - start_time

            print('Iter', epoch_i, ': Acc', acc, ', nmi', nmi, ', ari', ari, '; delta_label=', delta_label,";duration = %.3f" % duration,
                  "time_cost = %.3f" % time_cost,";KL loss = " ,kl_value)
            # print( 'Total loss=', total_value, '; MSE loss=', mse_value,";KL loss = " ,kl_value)
            # clustering(sess, X, para, mnist_input_images, mnist_input_labels)
            # clustering_batch(sess, X, para['encoded'], mnist_input_images, mnist_input_labels, batch_size)

            # check stop criterion
            if epoch_i > 0 and delta_label < tol:
                print('delta_label ', delta_label, '< tol ', tol)
                print('Reached tolerance threshold. Stopping training.')
                clustering_batch(sess, X, para['encoded'], mnist_input_images, mnist_input_labels, batch_size)
                break

            # cent_value = sess.run(centroides)
            # print(cent_value)
        global_value, rate = sess.run([global_step, learning_rate_dec])
        train,index = shuffle_dec_input.next_batch(batch_size)
        train_input_image = np.reshape(train, [len(train), 28, 28, 1])
        _, total_value,mse_value, kl_value = sess.run([optimizer_dec, total_loss,mse_loss,kl_loss], feed_dict={X: train_input_image, p_x: p_value[index,:]})
        # loss = kl_value
        # if (epoch_i + 1) % dec_display_step == 0:
        #     print('global_value=', global_value, '   ', 'rate = ', rate)
        #     time_cost = time.time() - start_time - duration
        #     duration = time.time() - start_time
        #     print("Epoch:", '%04d' % (epoch_i + 1), "kl_value=", "{:.9f}".format(kl_value), "duration = %.3f" % duration,
        #           "time_cost = %.3f" % time_cost)
    #################################重构样本与原始样本做对比###########################################
    # para_value = sess.run(para_z,feed_dict={X_layer1:mnist_input_images})
    # recon_X = para_value['output']
    # n_examples = 10
    # fig, axs = plt.subplots(2, n_examples, figsize=(10, 2))
    # for example_i in range(n_examples):
    #     axs[0][example_i].imshow(np.reshape(mnist_input_images[example_i, :], (28, 28)))
    #     axs[1][example_i].imshow(np.reshape(recon_X[example_i, :], (28, 28)))
    # fig.show()
    # plt.draw()
    # plt.waitforbuttonpress()

    #################################聚类###########################################
    # clustering_batch(sess, X, para['encoded'], mnist_input_images, mnist_input_labels, batch_size)
    # save_dir = r".\t-SNE"
    #
    # filename = 'mnist_cae.mat'
    # file_output = os.path.join(save_dir, filename)
    #
    # embedding_feature = feat_batch(sess, X, para['encoded'], mnist_input_images, batch_size)
    # sio.savemat(file_output, {'embedding_feature': embedding_feature})
    sess.close()


if __name__ == '__main__':
    test_mnist()