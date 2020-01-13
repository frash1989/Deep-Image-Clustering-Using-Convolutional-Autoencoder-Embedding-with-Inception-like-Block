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
from dataset_batch_process import *
from sklearn import preprocessing
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


def myAutoEncoder_dict(data_in):
    encoded1 = tf.layers.dense(data_in, units=500, activation=tf.nn.relu, name='encoded1')
    encoded2 = tf.layers.dense(encoded1, units=500, activation=tf.nn.relu, name='encoded2')
    encoded3 = tf.layers.dense(encoded2, units=2000, activation=tf.nn.relu, name='encoded3')
    embedding = tf.layers.dense(encoded3, units=10, activation=None, name='embedding')
    decoded1 = tf.layers.dense(embedding, units=2000, activation=tf.nn.relu, name='decoded1')
    decoded2 = tf.layers.dense(decoded1, units=500, activation=tf.nn.relu, name='decoded2')
    decoded3 = tf.layers.dense(decoded2, units=500, activation=tf.nn.relu, name='decoded3')
    reconstruct = tf.layers.dense(decoded3, units=data_in.get_shape()[1].value, activation=None, name='reconstruct')

    parameters = {'input': data_in,
                  'encoded1': encoded1,
                  'encoded2': encoded2,
                  'encoded3': encoded3,
                  'embedding': embedding,
                  'decoded1':decoded1,
                  'decoded2':decoded2,
                  'decoded3':decoded3,
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
    weight = q ** 2 / tf.reduce_sum(q, 0)
    return tf.transpose(tf.transpose(weight) / tf.reduce_sum(weight, 1))

def target_distribution_numpy(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T

def clustering(sess, net_in,net_out,data,label):
    #sess: session
    #net_in: placeholder
    #net_out: 网络的输出
    #data，label表示输入数据的情况，label是one_hot编码的
    #################################聚类###########################################
    net_output = sess.run(net_out['embedding'], feed_dict={net_in: data})
    start_time = time.time()
    kmeans = KMeans(n_clusters=10, n_init=20)  # n_clusters表示聚类的个数，n_init用不同的初始化质心运行算法的次数，默认为10
    y_pred = kmeans.fit_predict(net_output)
    y = label
    ACC = cluster_acc(y, y_pred)
    NMI = normalized_mutual_info_score(y, y_pred)
    print('ACC = %.2f , NMI = %.2f' % (ACC * 100, NMI * 100))

def main():
    # default = os.path.join(os.getenv('TEST_TMPDIR', '.'),
    #                        '../MNIST_data/'),
    mnist = input_data.read_data_sets('E:\TensorFlow\MyWork2017\MNIST_data', one_hot=True)
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

    input_labels = mnist_input_labels.argmax(1)
    y=input_labels
    ######################Kmeans of original data#######################
    # start_time = time.time()
    # mnist_kmeans = KMeans(n_clusters=10, n_init=20)  # n_clusters表示聚类的个数，n_init用不同的初始化质心运行算法的次数，默认为10
    # y_pred = mnist_kmeans.fit_predict(mnist_input_images)
    # duration = time.time() - start_time
    # print("Runtime of mnist  kmeans clustering = %.3f" % duration)
    # ACC = cluster_acc(y, y_pred)
    # NMI = normalized_mutual_info_score(y, y_pred)
    # print('Result of pca kmeans on mnist : ACC = %.2f   NMI = %.2f' % (ACC * 100, NMI * 100))
    ###################parameter setting##########################
    batch_size = 256
    n_epochs_finetune = 100000
    initial_learning_rate = 1e-3 # 初始学习率
    display_step = 1000
    cluster_display_step =10000
    n_input = mnist_total_images.shape[1]  # data input (feature shape)
    embedding_dim =10
    ##==================================================
    X = tf.placeholder("float", [None, n_input])
    global_step = tf.Variable(0,trainable=False)
    ##====================final clustering=============================
    para = myAutoEncoder_dict(X)

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
                                               decay_steps=50000,
                                               decay_rate=0.1,
                                               staircase=True)
    # MSE loss
    mse_loss= tf.reduce_mean(tf.square(tf.subtract(X , para['reconstruct'])))

    regularizer = tf.contrib.layers.l2_regularizer(scale=1e-4)
    reg_list = ['encoded1/kernel:0', 'encoded1/bias:0',
               'encoded2/kernel:0', 'encoded2/bias:0',
               'encoded3/kernel:0', 'encoded3/bias:0',
               'embedding/kernel:0', 'embedding/bias:0',
               'decoded1/kernel:0', 'decoded1/bias:0',
               'decoded2/kernel:0','decoded2/bias:0',
               'decoded3/kernel:0','decoded3/bias:0',
               'reconstruct/kernel:0','reconstruct/bias:0'
               ]


    reg_var = [v for v in tf.all_variables() if v.name in reg_list]  # Keep only the var
    reg_term = tf.contrib.layers.apply_regularization(regularizer, reg_var)

    total_cost =  mse_loss   #+  ireg_term

    # RMSE loss
    # total_cost = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(X_layer1, para['output']))))
    #Cross Entropy
    #Note: cross-entropy should only be used when the values are between 0 and 1.
    # epsilon =1e-10
    # cross_entropy = -1. * X_layer1 * tf.log(para['output']+epsilon) - (1. - X_layer1) * tf.log(1. - para['output']+epsilon)
    # total_cost = tf.reduce_mean(tf.reduce_sum(cross_entropy,reduction_indices=[1]))
    # total_cost = -tf.reduce_mean(tf.reduce_sum(X_layer1 * tf.log(para['output']), reduction_indices=[1]))

    optimizer_total = tf.train.AdamOptimizer(learning_rate).minimize(total_cost)
    # optimizer_total = tf.train.AdamOptimizer(learning_rate).minimize(total_cost,global_step=global_step)
    ############################Details of DEC###############################
    dec_epoch=10000000
    dec_display_step=1000
    dec_cluster_display_step=1000
    num_clusters = 10
    update_interval =  10#1*(len(mnist_input_labels)//batch_size +1)#10000#
    tol             = 1e-3

    centroides = tf.Variable(tf.random_normal([num_clusters, embedding_dim], stddev=0.01))
    q = confidence_assign(para['embedding'], centroides)
    p_x = tf.placeholder("float", [None, embedding_dim])
    kl_loss = kl_divergence(p_x, q)

    initial_learning_rate_dec = 1e-4

    learning_rate_dec = tf.train.exponential_decay(learning_rate=initial_learning_rate_dec,
                                               global_step=global_step,
                                               decay_steps=10000,
                                               decay_rate=0.9,
                                               staircase=True)

    optimizer_dec = tf.train.AdamOptimizer(learning_rate_dec).minimize(kl_loss)
    # optimizer_dec = tf.train.MomentumOptimizer(learning_rate_dec,0.9).minimize(kl_loss)
    # optimizer_dec = tf.train.AdamOptimizer(learning_rate_dec).minimize(kl_loss,global_step=global_step)
    all_variables = tf.all_variables()
    restore =  ['encoded1/kernel:0', 'encoded1/bias:0',
               'encoded2/kernel:0', 'encoded2/bias:0',
               'encoded3/kernel:0', 'encoded3/bias:0',
               'embedding/kernel:0', 'embedding/bias:0',
               'decoded1/kernel:0', 'decoded1/bias:0',
               'decoded2/kernel:0','decoded2/bias:0',
               'decoded3/kernel:0','decoded3/bias:0',
               'reconstruct/kernel:0','reconstruct/bias:0'
               ]

    restore_var = [v for v in tf.all_variables() if v.name in restore]  # Keep only the var
    print(restore_var)
    # help(tf.train.Saver)
    all_saver = tf.train.Saver(restore_var)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    #########################################################################################
    ckpt_dir = '../trained_models/mnist_full/DEC'
    # ckpt_dir = '../trained_models/mnist_test/DEC'
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        all_saver.restore(sess, ckpt.model_checkpoint_path)
        print('Restoring saved parameters!')
        clustering(sess, X, para, mnist_input_images, input_labels)
    else:
        print('Train networks!')
        start_time = time.time()
        duration = 0
        for epoch_i in range(n_epochs_finetune):
            global_value, rate = sess.run([global_step, learning_rate])
            train,train_index = shuffle_dec_input.next_batch(batch_size)
            _, c_total,mse_value = sess.run([optimizer_total, total_cost,mse_loss], feed_dict={X: train})
            if epoch_i == 0:
                print('global_value=', global_value, '   ', 'rate = ', rate)
                time_cost = time.time() - start_time - duration
                duration = time.time() - start_time
                print("Epoch:", '%04d' % (epoch_i + 1), "cost=", "{:.9f}".format(c_total), "duration = %.3f" % duration, "time_cost = %.3f" % time_cost,"mse_loss =", "{:.9f}".format(mse_value))
                # print("mse_loss =", "{:.9f}".format(mse_value), "reg_value = ""{:.9f}".format(regular_value) )
                print('**')
            if (epoch_i+1) % display_step == 0:
                # print('global_value=', global_value, '   ', 'rate = ', rate)
                time_cost = time.time() - start_time - duration
                duration = time.time() - start_time
                print("Epoch:", '%04d' % (epoch_i + 1), "cost=", "{:.9f}".format(c_total), "duration = %.3f" % duration, "time_cost = %.3f" % time_cost,"mse_loss =", "{:.9f}".format(mse_value))
                # print("mse_loss =", "{:.9f}".format(mse_value),
                #       "reg_value = ""{:.9f}".format(regular_value) )
            if (epoch_i + 1) % cluster_display_step == 0:
                clustering(sess, X, para, mnist_input_images, input_labels)

        clustering(sess, X, para, mnist_input_images, input_labels)

        print("Total Optimization Finished!")

        # Save the variables to disk.
        save_path_all = all_saver.save(sess, "../trained_models/mnist_full/DEC/dec_model.ckpt")
        # save_path_all = all_saver.save(sess, "../trained_models/mnist_test/DEC/dec_model.ckpt")
        print("All Model saved in file: ", save_path_all)


    save_dir = r"E:\TensorFlow\MyWork2017\t-SNE\embedding_features"

    filename = 'mnist_ae_kmeans.mat'
    file_output = os.path.join(save_dir, filename)
    embedding_feature = sess.run(para['embedding'], feed_dict={X: mnist_input_images})
    sio.savemat(file_output, {'embedding_feature': embedding_feature})

    print('Train DEC networks!')
    print('centroids initialization !')
    embedding_feature = sess.run(para['embedding'], feed_dict={X: mnist_input_images})
    kmeans = KMeans(n_clusters=10, n_init=20)  # n_clusters表示聚类的个数，n_init用不同的初始化质心运行算法的次数，默认为10
    y_pred = kmeans.fit_predict(embedding_feature)
    y_pred_last = y_pred
    temp_cent2 = sess.run(tf.assign(centroides, kmeans.cluster_centers_))

    loss   = 0
    start_time = time.time()
    duration = 0
    for epoch_i in range(dec_epoch):
        if epoch_i % update_interval==0:
            q_value = sess.run(q, feed_dict={X: mnist_input_images})
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

            print('Iter', epoch_i, ': Acc', acc, ', nmi', nmi, ', ari', ari, '; loss=', loss, '; delta_label=', delta_label,";duration = %.3f" % duration,
                  "time_cost = %.3f" % time_cost)
            clustering(sess, X, para, mnist_input_images, input_labels)

            # check stop criterion
            if epoch_i > 0 and delta_label < tol:
                print('delta_label ', delta_label, '< tol ', tol)
                print('Reached tolerance threshold. Stopping training.')
                break


        global_value, rate = sess.run([global_step, learning_rate_dec])
        train,index = shuffle_dec_input.next_batch(batch_size)
        _, kl_value = sess.run([optimizer_dec, kl_loss], feed_dict={X: train, p_x: p_value[index,:]})
        loss = kl_value
        if (epoch_i + 1) % dec_display_step == 0:
            print('global_value=', global_value, '   ', 'rate = ', rate)
            time_cost = time.time() - start_time - duration
            duration = time.time() - start_time
            print("Epoch:", '%04d' % (epoch_i + 1), "kl_value=", "{:.9f}".format(kl_value), "duration = %.3f" % duration,
                  "time_cost = %.3f" % time_cost)


    # save_dir = r".\t-SNE"
    #
    # filename = 'mnist_dec.mat'
    # file_output = os.path.join(save_dir, filename)
    #
    # embedding_feature = sess.run(para['embedding'], feed_dict={X: mnist_input_images})
    # sio.savemat(file_output, {'embedding_feature': embedding_feature})
    sess.close()


if __name__ == '__main__':
    main()