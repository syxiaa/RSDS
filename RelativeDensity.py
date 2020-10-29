from sklearn.ensemble import RandomForestClassifier
from random import choice
import pandas as pd
import heapq
from sklearn.utils import shuffle
from numpy import *
import warnings

warnings.filterwarnings("ignore")


def add_flip_noise(dataset, noise_rate):
    label_cat = list(set(dataset[:, 0]))
    new_data = array([])
    flag = 0
    for i in range(len(label_cat)):
        label = label_cat[i]
        other_label = list(filter(lambda x: x != label, label_cat))
        data = dataset[dataset[:, 0] == label]
        n = data.shape[0]
        noise_num = int(n * noise_rate)
        noise_index_list = []
        n_index = 0
        while True:
            rand_index = int(random.uniform(0, n))
            if rand_index in noise_index_list:
                continue
            if n_index < noise_num:
                data[rand_index, 0] = choice(other_label)  # todo
                n_index += 1
                noise_index_list.append(rand_index)
            if n_index >= noise_num:
                break
        if flag == 0:
            new_data = data
            flag = 1
        else:
            new_data = vstack([new_data, data])
    return new_data


def load_data(data, noise_rate):
    if noise_rate == 0:
        return data
    train = add_flip_noise(data, noise_rate)
    return train


def Get_n(train_data):
    data = train_data.values
    n = list(set(data[:, 0]))
    return n


def homo(train_data, k):
    n = Get_n(train_data)
    homo_list = pd.DataFrame(columns=train_data.columns)
    for a in n:
        data, maxvalue = train_data[train_data.loc[:, 0] == a], 1000
        dis, numpy_data = [[0] * len(data) for _ in range(len(data))], data.values
        for i in range(len(data)):
            dis[i][i] = maxvalue
            for j in range(i + 1, len(data)):
                dis[i][j] = np.linalg.norm(numpy_data[i, 1:] - numpy_data[j, 1:])
                dis[j][i] = dis[i][j]
        data['homo'] = list(map(lambda x: sum(heapq.nsmallest(k, x)) / k, dis))
        homo_list = homo_list.append(data)
    return homo_list


def hete(data, k):
    n = Get_n(data)
    hete_list = pd.DataFrame(columns=data.columns)
    for a in n:
        data_1, data_2 = data[data.loc[:, 0] != a], data[data.loc[:, 0] == a]
        numpy_data1, numpy_data2 = data_1.values, data_2.values
        dis = [[0] * len(data_1) for _ in range(len(data_2))]
        for i in range(len(data_2)):
            for j in range(len(data_1)):
                dis[i][j] = np.linalg.norm(
                    numpy_data2[i, 1:len(data.columns) - 1] - numpy_data1[j, 1:len(data.columns) - 1])
        data_2['hete'] = list(map(lambda x: sum(heapq.nsmallest(k, x)) / k, dis))
        hete_list = hete_list.append(data_2)
    return hete_list


def relativeDensity(train_data, k=3, p=1):
    homo_data = homo(train_data, k)
    df = hete(homo_data, k)
    df['relative'] = df['homo'] / df['hete']
    df = df[df['relative'] < p]
    df = df.drop(['homo', 'hete', 'relative'], axis=1)
    df = df.astype(float)
    df = df.values
    return df


def relativeDensity_smote(train_data, k=3, p=1):
    noise_df = []
    homo_data = homo(train_data, k)
    df = hete(homo_data, k)
    df['relative'] = df['homo'] / df['hete']
    for i in range(len(df)):
        if df['relative'][i] >= p:
            noise_df.append(i)
    df = df[df['relative'] < p]
    df = df.drop(['homo', 'hete', 'relative'], axis=1)
    df = df.astype(float)
    df = df.values
    return df, noise_df


def rf_mode(train_data, test_data):
    train_data, test_data = train_data.values, test_data.values
    train_label, train_data = train_data[:, 0], train_data[:, 1:]
    test_label, test_data = test_data[:, 0], test_data[:, 1:]
    model = RandomForestClassifier()
    model.fit(train_data, train_label)
    return model.score(test_data, test_label)


if __name__ == '__main__':
    df = pd.read_csv(r'D:\project\CRF\code\iris.csv')
    df = shuffle(df)
    df = df.reset_index(drop=True)
    train_data = df.loc[0:len(df) * 0.8, :].reset_index(drop=True)
    test_data = df.loc[len(df) * 0.8:, :].reset_index(drop=True)
    rd_train = relativeDensity(train_data)
    print(rd_train)
