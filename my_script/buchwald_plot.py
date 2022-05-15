import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# plt.style.use('ggplot')


df = pd.read_excel('/data/baiqing/PycharmProjects/CMPNN-master-concat_1/stat/Buchwald.xlsx', sheet_name='Sheet2')

# df_fullcv = df[:10]
# print(df_fullcv.columns)

fig = plt.figure(figsize=(15, 10))
sns.lineplot(x='Split Type', y='R2', data=df, hue='FP Type', style='FP Type', linewidth=3)

font_label = {'family': 'Nimbus Roman',
              'weight': 'bold',
              'style': 'normal',
              'size': 20}


# sns.lineplot(x='Split Type', y='R2 (GraphRXN--Sum)', data=df_fullcv)
# sns.lineplot(x='Split Type', y='R2 (GraphRXN--Concatenate)', data=df_fullcv)
# sns.lineplot(x='Split Type', y='R2 (BERT)', data=df_fullcv)
plt.xlabel('Random Seed', fontdict=font_label)
plt.ylabel(r'$R^{2}$', fontdict=font_label)
plt.xticks(fontsize=10, weight='bold')
plt.yticks(fontsize=15, weight='bold')
plt.savefig('../picture/Buchwald.png', dpi=300)
plt.show()


print('Done')
