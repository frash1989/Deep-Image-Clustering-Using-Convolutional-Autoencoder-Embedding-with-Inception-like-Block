import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys
import os
import cv2
import urllib
import random
import time
import scipy.io as sio
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn import metrics
import tensorflow.examples.tutorials.mnist.input_data as input_data
import h5py
from dataset_batch_process import *
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
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



def mnist_pca_clustering():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
    print("Train samples of mnist %d" % mnist.train.num_examples)
    print("Validation samples of mnist %d" % mnist.validation.num_examples)
    print("Test samples of mnist %d" % mnist.test.num_examples)

    print(type(mnist.train.images))
    print(len(mnist.train.images))  # 行数
    print(mnist.train.images.size)  # 元素个数
    print(mnist.train.images.shape)  # 行数，列数
    print(len(mnist.test.images))  # 行数
    print(mnist.test.images.size)  # 元素个数
    print(mnist.test.images.shape)  # 行数，列数

    mnist_total_images =np.vstack((mnist.train.images,mnist.validation.images,mnist.test.images))
    mnist_total_labels = np.squeeze(np.concatenate((mnist.train.labels, mnist.validation.labels, mnist.test.labels)))

    start_time = time.time()
    pca_mnist_full = PCA(n_components =10)
    mnist_full_pca = pca_mnist_full.fit_transform(mnist_total_images)
    mnist_kmeans = KMeans(n_clusters=10, n_init=20)#n_clusters表示聚类的个数，n_init用不同的初始化质心运行算法的次数，默认为10
    y_pred = mnist_kmeans.fit_predict(mnist_full_pca)
    duration = time.time() - start_time
    print("Runtime of mnist full kmeans clustering = %.3f" % duration)

    y = mnist_total_labels
    ACC = cluster_acc(y, y_pred)
    NMI = normalized_mutual_info_score(y, y_pred)
    print('Result of pca kmeans on mnist_full : ACC = %.2f   NMI = %.2f' % (ACC * 100, NMI * 100))


    start_time = time.time()
    pca_mnist_test = PCA(n_components =10)
    mnist_test_pca = pca_mnist_test.fit_transform(mnist.test.images)
    mnist_test_kmeans = KMeans(n_clusters=10, n_init=20)#n_clusters表示聚类的个数，n_init用不同的初始化质心运行算法的次数，默认为10
    y_pred = mnist_test_kmeans.fit_predict(mnist_test_pca)
    duration = time.time() - start_time
    print("Runtime of mnist test kmeans clustering = %.3f" % duration)

    y = mnist.test.labels
    ACC = cluster_acc(y, y_pred)
    NMI = normalized_mutual_info_score(y, y_pred)
    print('Result of pca kmeans on mnist_test : ACC = %.2f   NMI = %.2f' % (ACC * 100, NMI * 100))

def cifar_pca_clustering():
    data_path = r'F:\Dataset\cifar-10-batches-mat'
    test_file_name = 'test_batch.mat'
    full_file_name = 'cifar_full.mat'
    # file_name = 'USPS_11000.mat'


    # 读取matlab中非-v7.3格式存储的文件
    temp = sio.loadmat(os.path.join(data_path, full_file_name))
    full_labels = temp['labels']  # labels
    full_data = temp['data']  # labels
    full_data = full_data.astype('float32') / 255.0
    full_labels = np.squeeze(full_labels)

    start_time = time.time()
    pca_cifar_full = PCA(n_components =10)
    cifar_full_pca = pca_cifar_full.fit_transform(full_data)
    cifar_kmeans = KMeans(n_clusters=10, n_init=20)#n_clusters表示聚类的个数，n_init用不同的初始化质心运行算法的次数，默认为10
    y_pred = cifar_kmeans.fit_predict(cifar_full_pca)
    duration = time.time() - start_time
    print("Runtime of cifar full kmeans clustering = %.3f" % duration)

    y = full_labels
    ACC = cluster_acc(y, y_pred)
    NMI = normalized_mutual_info_score(y, y_pred)
    print('Result of pca kmeans on cifar_full : ACC = %.2f   NMI = %.2f' % (ACC * 100, NMI * 100))


    # 读取matlab中非-v7.3格式存储的文件
    temp = sio.loadmat(os.path.join(data_path, test_file_name))
    test_labels = temp['labels']  # labels
    test_data = temp['data']  # labels
    test_data = test_data.astype('float32') / 255.0
    test_labels = np.squeeze(test_labels)

    start_time = time.time()
    pca_cifar_test = PCA(n_components =10)
    cifar_test_pca = pca_cifar_test.fit_transform(test_data)
    cifar_test_kmeans = KMeans(n_clusters=10, n_init=20)#n_clusters表示聚类的个数，n_init用不同的初始化质心运行算法的次数，默认为10
    y_pred = cifar_test_kmeans.fit_predict(cifar_test_pca)
    duration = time.time() - start_time
    print("Runtime of cifar test kmeans clustering = %.3f" % duration)

    y = test_labels
    ACC = cluster_acc(y, y_pred)
    NMI = normalized_mutual_info_score(y, y_pred)
    print('Result of pca kmeans on cifar_test : ACC = %.2f   NMI = %.2f' % (ACC * 100, NMI * 100))
def usps_pca_clustering():
    data_path = r'F:\Dataset\USPS'
    file_name = 'USPS_11000.mat'


    # 读取matlab中非-v7.3格式存储的文件
    temp = sio.loadmat(os.path.join(data_path, file_name))
    full_labels = temp['labels']  # labels
    full_data = temp['data']  # labels
    # full_data = full_data.astype('float32') / 255.0
    full_data = (full_data - np.float32(127.5)) / np.float32(127.5)
    full_labels = np.squeeze(full_labels)

    start_time = time.time()
    pca_usps_full = PCA(n_components =10)
    usps_full_pca = pca_usps_full.fit_transform(full_data)
    usps_kmeans = KMeans(n_clusters=10, n_init=20)#n_clusters表示聚类的个数，n_init用不同的初始化质心运行算法的次数，默认为10
    y_pred = usps_kmeans.fit_predict(usps_full_pca)
    duration = time.time() - start_time
    print("Runtime of usps kmeans clustering = %.3f" % duration)

    y = full_labels
    ACC = cluster_acc(y, y_pred)
    NMI = normalized_mutual_info_score(y, y_pred)
    print('Result of pca kmeans on usps : ACC = %.2f   NMI = %.2f' % (ACC * 100, NMI * 100))
def stl_pca_clustering():
    data_path = r'F:\Dataset\stl10_matlab'
    file_name = 'STL_10.mat'



    # 读取matlab中非-v7.3格式存储的文件
    temp = sio.loadmat(os.path.join(data_path, file_name))
    full_labels = temp['labels']  # labels
    full_data = temp['data']  # labels
    full_data = full_data.astype('float32') / 255.0
    full_labels = np.squeeze(full_labels)

    start_time = time.time()
    pca_stl_full = PCA(n_components =10)
    stl_full_pca = pca_stl_full.fit_transform(full_data)
    stl_kmeans = KMeans(n_clusters=10, n_init=20)#n_clusters表示聚类的个数，n_init用不同的初始化质心运行算法的次数，默认为10
    y_pred = stl_kmeans.fit_predict(stl_full_pca)
    duration = time.time() - start_time
    print("Runtime of STL kmeans clustering = %.3f" % duration)

    y = full_labels
    ACC = cluster_acc(y, y_pred)
    NMI = normalized_mutual_info_score(y, y_pred)
    print('Result of pca kmeans on STL : ACC = %.2f   NMI = %.2f' % (ACC * 100, NMI * 100))
def mnist_sc():

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    print("Train samples of mnist %d" % mnist.train.num_examples)
    print("Validation samples of mnist %d" % mnist.validation.num_examples)
    print("Test samples of mnist %d" % mnist.test.num_examples)

    print(type(mnist.train.images))
    print(len(mnist.train.images))  # 行数
    print(mnist.train.images.size)  # 元素个数
    print(mnist.train.images.shape)  # 行数，列数

    mnist_images = np.vstack((mnist.train.images, mnist.validation.images, mnist.test.images))
    mnist_labels = np.vstack((mnist.train.labels, mnist.validation.labels, mnist.test.labels))

    data = mnist_images
    labels = mnist_labels.argmax(1)
    num_clusters = len(np.unique(labels))
    #################################聚类###########################################
    start_time = time.time()
    gamma =1
    y_pred = SpectralClustering(n_clusters=num_clusters, gamma=gamma).fit_predict(data)
    duration = time.time() - start_time
    print("Runtime of mnist full spectral clustering = %.3f" % duration)
    y = labels
    ACC = cluster_acc(y, y_pred)
    NMI = normalized_mutual_info_score(y, y_pred)
    print('Result of spectral clusteirng on mnist_full : ACC = %.2f   NMI = %.2f' % (ACC * 100, NMI * 100))
    #################################T-SNE展示###########################################

def cifar_sc():
    data_path = r'F:\Dataset\cifar-10-batches-mat'
    # test_file_name = 'test_batch.mat'
    full_file_name = 'cifar_full.mat'
    # 读取matlab中非-v7.3格式存储的文件
    temp = sio.loadmat(os.path.join(data_path, full_file_name))
    full_labels = temp['labels']  # labels
    full_data = temp['data']  # labels
    full_data = full_data.astype('float32') / 255.0
    full_labels = np.squeeze(full_labels)

    data = full_data
    labels = full_labels
    num_clusters = len(np.unique(labels))
    #################################聚类###########################################
    start_time = time.time()
    gamma =1
    y_pred = SpectralClustering(n_clusters=num_clusters, gamma=gamma).fit_predict(data)
    duration = time.time() - start_time
    print("Runtime of cifar full spectral clustering = %.3f" % duration)
    y = labels
    ACC = cluster_acc(y, y_pred)
    NMI = normalized_mutual_info_score(y, y_pred)
    print('Result of spectral clusteirng on cifar_full : ACC = %.2f   NMI = %.2f' % (ACC * 100, NMI * 100))
    #################################T-SNE展示###########################################
def usps_sc():
    data_path = r'F:\Dataset\USPS'
    file_name = 'USPS_11000.mat'

    # 读取matlab中非-v7.3格式存储的文件
    temp = sio.loadmat(os.path.join(data_path, file_name))
    full_labels = temp['labels']  # labels
    full_data = temp['data']  # labels
    full_data = full_data.astype('float32') / 255.0
    # full_data = (full_data - np.float32(127.5)) / np.float32(127.5)
    full_labels = np.squeeze(full_labels)

    data = full_data
    labels = full_labels
    num_clusters = len(np.unique(labels))
    #################################聚类###########################################
    start_time = time.time()
    gamma =1
    y_pred = SpectralClustering(n_clusters=num_clusters, gamma=gamma).fit_predict(data)
    duration = time.time() - start_time
    print("Runtime of usps spectral clustering = %.3f" % duration)
    y = labels
    ACC = cluster_acc(y, y_pred)
    NMI = normalized_mutual_info_score(y, y_pred)
    print('Result of spectral clusteirng on usps : ACC = %.2f   NMI = %.2f' % (ACC * 100, NMI * 100))
    #################################T-SNE展示###########################################
def stl_sc():
    data_path = r'F:\Dataset\stl10_matlab'
    file_name = 'STL_10.mat'

    # 读取matlab中非-v7.3格式存储的文件
    temp = sio.loadmat(os.path.join(data_path, file_name))
    full_labels = temp['labels']  # labels
    full_data = temp['data']  # labels
    full_data = full_data.astype('float32') / 255.0
    full_labels = np.squeeze(full_labels)

    data = full_data
    labels = full_labels
    num_clusters = len(np.unique(labels))
    #################################聚类###########################################
    start_time = time.time()
    gamma =1
    y_pred = SpectralClustering(n_clusters=num_clusters, gamma=gamma).fit_predict(data)
    duration = time.time() - start_time
    print("Runtime of stl spectral clustering = %.3f" % duration)
    y = labels
    ACC = cluster_acc(y, y_pred)
    NMI = normalized_mutual_info_score(y, y_pred)
    print('Result of spectral clusteirng on stl : ACC = %.2f   NMI = %.2f' % (ACC * 100, NMI * 100))
    #################################T-SNE展示###########################################
def main():
    #设定随机数生成序列种子，从而使得每次的结果可复现
    # mnist_pca_clustering()
    # cifar_pca_clustering()
    # usps_pca_clustering()
    # stl_pca_clustering()

    usps_sc()
    stl_sc()
    cifar_sc()
    mnist_sc()






if __name__ == '__main__':
        main()
