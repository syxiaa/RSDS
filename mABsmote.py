from sklearn.neighbors import NearestNeighbors
from sklearn.utils import safe_indexing

from base_sampler import *

from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import Counter
import warnings

warnings.filterwarnings("ignore")


def make_circles_data(rate):
    X, labels = make_circles(n_samples=2000, noise=0.1, factor=rate, random_state=1)  # factor表示里圈和外圈的距离之比.
    # print("X.shape:", X.shape)
    # print("labels:", set(labels))
    # print(type(labels))
    data = np.hstack((labels.reshape(len(labels), 1), X))

    data_0 = data[data[:, 0] == 0]  # 0为多数类
    data_1 = data[data[:, 0] == 1]
    item = list(range(0, len(data_1)))
    random.shuffle(item)
    data_1 = data_1[item[:int(len(item) * 0.2)]]  # 10:2

    data = np.vstack((data_0, data_1))
    return data


def plot_ABsmote(data):
    """画原始数据的点"""
    labels = data[:, 0]
    X = data[:, 1:]
    unique_lables = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_lables)))
    for k, col in zip(unique_lables, colors):
        x_k = X[labels == k]
        plt.scatter(x_k[:, 0], x_k[:, 1], marker='o', s=5)  # , markerfacecolor=col, markeredgecolor="k", markersize=14
    plt.title('data by make_circles()')
    plt.show()


def plot_add_data(new_data, orignial_data, title=''):
    """画原始点和新加的点"""
    labels = orignial_data[:, 0]
    X = orignial_data[:, 1:]
    unique_lables = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_lables)))
    for k, col in zip(unique_lables, colors):
        x_k = X[labels == k]
        plt.scatter(x_k[:, 0], x_k[:, 1], marker='o', s=5,c=col)
    plt.scatter(new_data[:, 1], new_data[:, 2], marker='*', s=5)
    plt.title(title)
    plt.show()


# 处于多数类与少数类边缘的样本
def in_danger(imbalanced_featured_data, old_feature_data, old_label_data, imbalanced_label_data):
    nn_m = NearestNeighbors(n_neighbors=9).fit(imbalanced_featured_data)
    # 获取每一个少数类样本点周围最近的n_neighbors-1个点的位置矩阵
    nnm_x = NearestNeighbors(n_neighbors=9).fit(imbalanced_featured_data).kneighbors(old_feature_data,
                                                                                     return_distance=False)[:, 1:]
    nn_label = (imbalanced_label_data[nnm_x] != old_label_data).astype(int)
    n_maj = np.sum(nn_label, axis=1)
    return np.bitwise_and(n_maj >= (nn_m.n_neighbors - 1) / 2, n_maj < nn_m.n_neighbors - 1)


def make_sample(imbalanced_data_arr2, diff):
    # 将数据集分开为少数类数据和多数类数据
    minor_data_arr2, major_data_arr2 = seperate_minor_and_major_data(imbalanced_data_arr2)
    imbalanced_featured_data = imbalanced_data_arr2[:, 1:]
    imbalanced_label_data = imbalanced_data_arr2[:, 0]
    # 原始少数样本的特征集
    old_feature_data = minor_data_arr2[:, 1:]
    # 原始少数样本的标签值
    old_label_data = minor_data_arr2[0][0]
    danger_index = in_danger(imbalanced_featured_data, old_feature_data, old_label_data, imbalanced_label_data)
    # 少数样本中噪音集合，也就是最终要产生新样本的集合
    danger_index_data = safe_indexing(old_feature_data, danger_index)
    # danger_feature_data=imbalanced_data_arr2[danger_index_data]
    # 获取每一个少数类样本点周围最近的n_neighbors-1个点的位置矩阵
    nns = NearestNeighbors(n_neighbors=6).fit(old_feature_data).kneighbors(danger_index_data, return_distance=False)[:,
          1:]  # [:, 1:] 为了去掉本身的索引
    # 随机产生diff个随机数作为之后产生新样本的选取的样本下标值
    samples_indices = np.random.randint(low=0, high=np.shape(danger_index_data)[0], size=diff)
    # 随机产生diff个随机数作为之后产生新样本的间距值
    steps = np.random.uniform(size=diff)
    cols = np.mod(samples_indices, nns.shape[1])
    reshaped_feature = np.zeros((diff, danger_index_data.shape[1]))
    for i, (col, step) in enumerate(zip(cols, steps)):
        row = samples_indices[i]
        reshaped_feature[i] = danger_index_data[row] - step * (danger_index_data[row] - old_feature_data[
            nns[row, col]])
    new_min_feature_data = np.vstack((reshaped_feature, old_feature_data))
    return new_min_feature_data


# 产生少数类新样本的方法
def make_sample_AB(imbalanced_data_arr2, diff):
    # 将数据集分开为少数类数据和多数类数据
    minor_data_arr2, major_data_arr2 = seperate_minor_and_major_data(imbalanced_data_arr2)
    imbalanced_featured_data = imbalanced_data_arr2[:, 1:]
    imbalanced_label_data = imbalanced_data_arr2[:, 0]
    # 原始少数样本的特征集
    old_feature_data = minor_data_arr2[:, 1:]
    # 原始少数样本的标签值
    old_label_data = minor_data_arr2[0][0]
    danger_index = in_danger(imbalanced_featured_data, old_feature_data, old_label_data, imbalanced_label_data)
    # 少数样本中噪音集合，也就是最终要产生新样本的集合
    danger_index_data = safe_indexing(old_feature_data, danger_index)
    # danger_feature_data=imbalanced_data_arr2[danger_index_data]
    # 获取每一个少数类样本点周围最近的n_neighbors-1个点的位置矩阵
    k = min(len(danger_index_data), 6)
    # print('k AB ',k)
    nns = NearestNeighbors(n_neighbors=k).fit(danger_index_data).kneighbors(danger_index_data, return_distance=False)[:,
          1:]  # [:, 1:] 为了去掉本身的索引
    # print(old_feature_data[0,:])
    # todo fit (old_feature_data) 被改为了 fit (danger_index_data)
    # 随机产生diff个随机数作为之后产生新样本的选取的样本下标值
    samples_indices = np.random.randint(low=0, high=np.shape(danger_index_data)[0], size=diff)
    # 随机产生diff个随机数作为之后产生新样本的间距值
    steps = np.random.uniform(size=diff)
    cols = np.mod(samples_indices, nns.shape[1])
    reshaped_feature = np.zeros((diff, danger_index_data.shape[1]))
    for i, (col, step) in enumerate(zip(cols, steps)):
        row = samples_indices[i]
        reshaped_feature[i] = danger_index_data[row] - step * (danger_index_data[row] - danger_index_data[
            nns[row, col]])  # todo 后面一个 danger_index_data 原来是 old_feature_data
    new_min_feature_data = np.vstack((old_feature_data, reshaped_feature))
    return new_min_feature_data


# 对不平衡的数据集imbalanced_data_arr2进行Border-SMOTE采样操作，返回平衡数据集
# :param imbalanced_data_arr2: 非平衡数据集
# :return: 平衡后的数据集
def Affinitive_Border_SMOTE(imbalanced_data_arr2):
    # 将数据集分开为少数类数据和多数类数据
    minor_data_arr2, major_data_arr2 = seperate_minor_and_major_data(imbalanced_data_arr2)
    # 计算多数类数据和少数类数据之间的数量差,也是需要过采样的数量
    diff = major_data_arr2.shape[0] - minor_data_arr2.shape[0]
    # 原始少数样本的标签值
    old_label_data = minor_data_arr2[0][0]  # 标签值
    # 使用K近邻方法产生的新样本特征集
    # new_feature_data = make_sample(imbalanced_data_arr2, diff)  # todo borderline
    new_feature_data_AB = make_sample_AB(imbalanced_data_arr2, diff)  # todo AB-smote改动的地方
    # 使用K近邻方法产生的新样本标签数组
    new_labels_data = np.array([old_label_data] * np.shape(major_data_arr2)[0])
    # 将类别标签数组合并到少数类样本特征集，构建出新的少数类样本数据集
    # new_minor_data_arr2 = np.column_stack((new_feature_data, new_labels_data))  # todo borderline
    new_minor_data_arr2_AB = np.column_stack((new_labels_data, new_feature_data_AB))
    # 将少数类数据集和多数据类数据集合并，并对样本数据进行打乱重排，
    # balanced_data_arr2 = concat_and_shuffle_data(new_minor_data_arr2_AB, major_data_arr2)
    balanced_data_arr2 = np.vstack((major_data_arr2, new_minor_data_arr2_AB))
    return balanced_data_arr2  # , new_minor_data_arr2  # todo borderline


def mABsmote_fun(data):
    counter = Counter(data[:, 0])

    # 找样本数最多的类别
    max_class_label, max_class_number = 0, 0
    for k, v in counter.items():
        if v > max_class_number:
            max_class_label, max_class_number = k, v

    data_new = np.array([]).reshape((-1, data.shape[1]))

    data_more = data[data[:, 0] == max_class_label, :]
    for k, v in counter.items():
        if v == max_class_number:
            continue
        data_less = data[data[:, 0] == k, :]
        data_train = np.vstack((data_more, data_less))

        data_absmote = Affinitive_Border_SMOTE(data_train)

        if data_new.shape[0] == 0:
            data_new = np.vstack((data_new, data_absmote))
        else:
            data_new = np.vstack((data_new, data_absmote[data_absmote[:, 0] != max_class_label, :]))

    return data_new


# 测试
if __name__ == '__main__':
    """
    Affinitive_Border_SMOTE 方法就是在Borderline-smote-1的基础上做了改进 
    在Borderline-smote-1中 让danger的点对所有少数类的点进行Knn插值
    而Affinitive_Border_SMOTE 则是让danger的内部点 进行Knn插值 
    此源码是Borderline-smote-1的 只是将函数Affinitive_Border_SMOTE 进行微调 
    """

    imbalanced_data = make_circles_data(0.6)            #numpy.ndarray

    # plot_ABsmote(imbalanced_data)
    # print(len(set(imbalanced_data[:, 0])))      #标签种数，第0列是标签
    # print('原始不平衡数据集', imbalanced_data.shape)
    # minor_data_arr2, major_data_arr2 = seperate_minor_and_major_data(imbalanced_data)
    # print('少数类', minor_data_arr2.shape)
    # print('多数类', major_data_arr2.shape,'\n')

    # 测试Affinitive_Border_SMOTE方法
    add_new_data_AB = Affinitive_Border_SMOTE(imbalanced_data)  # 采样后的数据（新加点加上原始数据） , minor_new_data
    minor_data_arr2, major_data_arr2 = seperate_minor_and_major_data(add_new_data_AB)
    # print('采样后的少数类', minor_data_arr2.shape)
    # print('采样后的多数类', major_data_arr2.shape)
    # print(add_new_data_AB.shape)
    # plot_add_data(minor_new_data, imbalanced_data, title='Smote-Borderline-1')  # todo borderline
    # plot_add_data(add_new_data_AB, imbalanced_data, title='ABsmote')  # 新加的点 原始的点
    # plot_ABsmote(minor_data_arr2)
    # plot_ABsmote(major_data_arr2)
    # plot_ABsmote(add_new_data_AB)

    # balanced_data_arr2 = concat_and_shuffle_data(add_new_data_AB, major_data_arr2)  # 所有数据加在一起 打乱
    # balanced_data_arr2, new_data = Affinitive_Border_SMOTE(imbalanced_data)
    # print('平衡数据集', balanced_data_arr2.shape)
