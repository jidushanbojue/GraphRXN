import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def plot_auc(logfile):
    df = pd.read_csv(logfile, names=['epoch', 'AUC'])
    fig = plt.figure(figsize=(20, 15), dpi=300)
    sns.lineplot(x='epoch', y='AUC', data=df, label='AUC', linewidth=5, color='red')
    plt.legend()
    plt.title('AUC of CMPNN model', fontsize=20)
    plt.ylabel('AUC', fontsize=20)
    plt.xlabel('Epoch', fontsize=20)
    plt.tick_params(labelsize=15)
    plt.savefig('CMPNN_AUC.png')
    # plt.show()
    print('Done')



if __name__ == '__main__':
    base_dir = '~/PycharmProjects/CMPNN-master/data'
    log_file = os.path.join(base_dir, 'verbose.log')
    plot_auc(logfile=log_file)
