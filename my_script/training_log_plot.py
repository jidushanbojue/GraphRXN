import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# plt.style.use('ggplot')
font_label = {'family': 'Nimbus Roman',
              'weight': 'bold',
              'style': 'normal',
              'size': 10}

title_label = {'family': 'Nimbus Roman',
              'weight': 'bold',
              'style': 'normal',
              'size': 50}

def log_plot(file_list):
    fig = plt.figure(figsize=(20, 20))

    for idx, file in enumerate(file_list, 1):
        # df = pd.read_csv(file).rename(columns={'idx': 'Epoch'})
        df = pd.read_csv(file)
        # number = int('52'+str(idx))
        plt.subplot(5, 1, idx)
        sns.lineplot(x='idx', y='R2_concat', data=df, linewidth=3, color='green', label='concat')
        sns.lineplot(x='idx', y='R2_sum', data=df, linewidth=3, color='red', label='sum')
        plt.xlabel('Epoch (Fold {})'.format(idx), fontdict=font_label)
        plt.ylabel(r'$R^{2}$', fontdict=font_label)
    fig.suptitle('The training log of validation set of 10-CV fold on in-house dataset', size=20)
    plt.savefig('/data/baiqing/PycharmProjects/GraphRXN/picture/in_house_training_log.png', dpi=300)
    plt.show()


    # df = pd.read_csv(file)
    # fig = plt.figure(figsize=(15, 10))
    # plt.subplot(521)
    # sns.lineplot(x='idx', y='R2_concat', data=df, linewidth=3, color='green', label='concat')
    # sns.lineplot(x='idx', y='R2_sum', data=df, linewidth=3, color='red', label='sum')
    # plt.xlabel('Epoch', fontdict=font_label)
    # plt.ylabel(r'$R^{2}$', fontdict=font_label)
    # plt.xticks(fontsize=10, weight='bold')
    # plt.yticks(fontsize=15, weight='bold')
    # plt.legend()
    #
    #
    # plt.subplot(522)
    # sns.lineplot(x='idx', y='R2_concat', data=df, linewidth=3, color='green', label='concat')
    # sns.lineplot(x='idx', y='R2_sum', data=df, linewidth=3, color='red', label='sum')
    # plt.xlabel('Epoch', fontdict=font_label)
    # plt.ylabel(r'$R^{2}$', fontdict=font_label)
    # plt.xticks(fontsize=10, weight='bold')
    # plt.yticks(fontsize=15, weight='bold')
    # plt.legend()

    plt.show()


# if __name__ == '__main__':
#     base_dir = '/data/baiqing/PycharmProjects/GraphRXN/stat'
#     file_list = []
#     idx_list = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']
#     for idx in idx_list:
#         # file_name = 'log_buchard_concat_sum_' + idx + '.csv'
#         # file_name = 'log_suzuki_concat_sum_' + idx + '.csv'
#         file_name = 'log_denmark_concat_sum_' + idx + '.csv'
#         file = os.path.join(base_dir, file_name)
#         file_list.append(file)
#     log_plot(file_list)

### For in-house
if __name__ == '__main__':
    base_dir = '/data/baiqing/PycharmProjects/GraphRXN/stat'
    file_list = []
    idx_list = ['01', '02', '03', '04', '05']
    for idx in idx_list:
        # file_name = 'log_buchard_concat_sum_' + idx + '.csv'
        # file_name = 'log_suzuki_concat_sum_' + idx + '.csv'
        file_name = 'log_in_house_concat_sum_' + idx + '.csv'
        file = os.path.join(base_dir, file_name)
        file_list.append(file)
    log_plot(file_list)





# df = pd.read_csv('/data/baiqing/PycharmProjects/GraphRXN/stat/log_buchard_concat_sum_01.csv')
#
# # df_fullcv = df[:10]
# # print(df_fullcv.columns)
#
# fig = plt.figure(figsize=(15, 10))
# sns.lineplot(x='idx', y='R2_concat', data=df, linewidth=3, color='green', label='concat')
# sns.lineplot(x='idx', y='R2_sum', data=df, linewidth=3, color='red', label='sum')
#
# font_label = {'family': 'Nimbus Roman',
#               'weight': 'bold',
#               'style': 'normal',
#               'size': 20}
#
#
# # sns.lineplot(x='Split Type', y='R2 (GraphRXN--Sum)', data=df_fullcv)
# # sns.lineplot(x='Split Type', y='R2 (GraphRXN--Concatenate)', data=df_fullcv)
# # sns.lineplot(x='Split Type', y='R2 (BERT)', data=df_fullcv)
# plt.xlabel('Epoch', fontdict=font_label)
# plt.ylabel(r'$R^{2}$', fontdict=font_label)
# plt.xticks(fontsize=10, weight='bold')
# plt.yticks(fontsize=15, weight='bold')
# plt.legend()
# # plt.savefig('../picture/Suzuki.png', dpi=300)
# plt.show()


print('Done')