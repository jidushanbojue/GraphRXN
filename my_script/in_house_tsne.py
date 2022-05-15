from sklearn.manifold import TSNE, MDS
import pandas as pd
import os

import seaborn as sns
from matplotlib import pyplot as plt


font = {'family' : 'serif',
        'color'  : 'darkred',
        'weight' : 'normal',
        'size'   : 20,
        }

def work(file):
    tsne = TSNE(n_components=2, init='pca', random_state=42)
    df = pd.read_csv(file)
    target = df['Output']
    Group = df['Group']
    ID = df['Products_Num']
    Bromide = df['Bromide']
    Amine = df['Amine']
    Product = df['Product']
    rsmi = df['rsmi']


    # G1 = Group.groupby('group')
    X = df.drop(['Output', 'Group', 'Products_Num', 'Bromide', 'Amine', 'Product', 'rsmi'], axis=1)

    X_tsne = tsne.fit_transform(X)

    # x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    # X_norm = (X_tsne-x_min)/(x_max-x_min)

    # plt.plot()
    # sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], style={})
    fig = plt.figure(figsize=(20, 15))

    plt.scatter(x=X_tsne[0:317, 0], y=X_tsne[0:317, 1], c=target[0:317], marker='*', s=200, cmap=plt.cm.bwr, label='G1')
    plt.scatter(x=X_tsne[317:736, 0], y=X_tsne[317:736, 1], c=target[317:736], marker='1', s=200, cmap=plt.cm.bwr, label='G2')
    plt.scatter(x=X_tsne[736:1137, 0], y=X_tsne[736:1137, 1], c=target[736:1137], marker='x', s=200, cmap=plt.cm.bwr, label='G3')
    plt.scatter(x=X_tsne[1137:, 0], y=X_tsne[1137:, 1], c=target[1137:], marker=10, s=200, cmap=plt.cm.bwr, label='G4')


    # plt.scatter(x=X_tsne[0:3956, 0], y=X_tsne[0:3956, 1], c=target[0:3956], marker='*', s=200, cmap=plt.cm.bwr, label='Buchwald')
    # plt.scatter(x=X_tsne[3956:9717, 0], y=X_tsne[3956:9717, 1], c=target[3956:9717], marker='1', s=200, cmap=plt.cm.bwr, label='Suzuki')
    # plt.scatter(x=X_tsne[9717:, 0], y=X_tsne[9717:, 1], c=target[9717:], marker=10, s=200, cmap=plt.cm.bwr, label='Denmark')

    # plt.scatter(x=X_tsne[0:3956, 0], y=X_tsne[0:3956, 1], marker='*', s=200, label='Buchwald', c='blue')
    # plt.scatter(x=X_tsne[3956:9717, 0], y=X_tsne[3956:9717, 1], marker='1', s=200, label='Suzuki', c='red')
    # plt.scatter(x=X_tsne[9717:, 0], y=X_tsne[9717:, 1], marker=10, s=200, label='Denmark', c='green')


    cb = plt.colorbar()
    cb.set_label('Reaction Output', fontdict=font)
    cb.ax.tick_params(labelsize=16)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=15)
    plt.savefig('in_house_dataset_sum.png', dpi=300)
    plt.show()

    print('Done')


def work_mds(file):
    mds = MDS(n_components=2)
    df = pd.read_csv(file)
    target = df['Output']
    Group = df['group']
    # G1 = Group.groupby('group')
    X = df.drop(['Output', 'group'], axis=1)

    X_mds = mds.fit_transform(X)

    x_min, x_max = X_mds.min(0), X_mds.max(0)
    X_norm = (X_mds-x_min)/(x_max-x_min)

    # plt.plot()
    # sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], style={})
    fig = plt.figure(figsize=(20, 15))

    plt.scatter(x=X_mds[0:200, 0], y=X_mds[0:200, 1], c=target[0:200], marker='*', s=200, cmap=plt.cm.bwr, label='G1')
    plt.scatter(x=X_mds[200:350, 0], y=X_mds[200:350, 1], c=target[200:350], marker='1', s=200, cmap=plt.cm.bwr, label='G2')
    plt.scatter(x=X_mds[350:560, 0], y=X_mds[350:560, 1], c=target[350:560], marker='x', s=200, cmap=plt.cm.bwr, label='G3')
    plt.scatter(x=X_mds[560:, 0], y=X_mds[560:, 1], c=target[560:], marker=10, s=200, cmap=plt.cm.bwr, label='G4')
    cb = plt.colorbar()
    cb.set_label('Yield', fontdict=font)
    cb.ax.tick_params(labelsize=16)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=15)
    plt.savefig('concat_mds.png', dpi=300)
    plt.show()

    print('Done')




if __name__ == '__main__':
    # base_dir = '/data/baiqing/PycharmProjects/GraphRXN/result/sushimin'
    base_dir = '/data/baiqing/PycharmProjects/GraphRXN/stat'
    # first = os.path.join(base_dir, 'concat_tsne.csv')
    first = os.path.join(base_dir, 'in_house_sum_encoder_tsne.csv')
    # first = os.path.join(base_dir, 'three_group_dataset_tsne_sum.csv')
    work(first)
    # work_mds(first)









