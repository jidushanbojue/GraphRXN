import os
from glob import glob
for i in range(0,11):
    print(i)
    cmd1 = f'python predict.py ' \
           f'--test_path data/denmark_0414/FullCV_0{i}_test_products_generated_temp.csv ' \
           f'--checkpoint_dir result/Denmark/concat_0{i}_temp_add_product/ ' \
           f'--preds_path result/Denmark/concat_0{i}_temp_add_product/preds.csv ' \
           f'--preds_stat_path result/Denmark/concat_0{i}_temp_add_product/stats.csv'

    cmd2 = f'python predict.py ' \
           f'--test_path data/denmark_0414/FullCV_0{i}_test_products_generated_temp.csv ' \
           f'--checkpoint_dir result/Denmark/sum_0{i}_temp_add_product/ ' \
           f'--preds_path result/Denmark/sum_0{i}_temp_add_product/preds.csv ' \
           f'--preds_stat_path result/Denmark/sum_0{i}_temp_add_product/stats.csv'

    print(cmd1)
    os.system(cmd1)
    print('-------------------------------------')
    print(cmd2)
    os.system(cmd2)


cmd1 = f'python predict.py ' \
       f'--test_path data/denmark_0414/FullCV_{10}_test_products_generated_temp.csv ' \
       f'--checkpoint_dir result/Denmark/concat_{10}_temp_add_product/ ' \
       f'--preds_path result/Denmark/concat_{10}_temp_add_product/preds.csv ' \
       f'--preds_stat_path result/Denmark/concat_{10}_temp_add_product/stats.csv'

cmd2 = f'python predict.py ' \
       f'--test_path data/denmark_0414/FullCV_{10}_test_products_generated_temp.csv ' \
       f'--checkpoint_dir result/Denmark/sum_{10}_temp_add_product/ ' \
       f'--preds_path result/Denmark/sum_{10}_temp_add_product/preds.csv ' \
       f'--preds_stat_path result/Denmark/sum_{10}_temp_add_product/stats.csv'

print(cmd1)
os.system(cmd1)
print('-------------------------------------')
print(cmd2)
os.system(cmd2)

pred_fnames = glob('/home/baiqing/PycharmProjects/GraphRXN/result/Denmark/*_temp_add_product/*preds.csv')
print(len(pred_fnames))
stats_fnames = glob('/home/baiqing/PycharmProjects/GraphRXN/result/Denmark/*_temp_add_product/*stats.csv')
print(len(stats_fnames))


from glob import glob
fnames = glob('/home/baiqing/PycharmProjects/GraphRXN/result/Buchwald/*_tsne/*.png')
save_path = '/home/baiqing/PycharmProjects/GraphRXN/result_png/Buchwald'

fnames = glob('/home/baiqing/PycharmProjects/GraphRXN/result/Denmark/*_temp_add_product/*.png')
save_path = '/home/baiqing/PycharmProjects/GraphRXN/result_png/Denmark'

'/home/baiqing/PycharmProjects/GraphRXN/result/Suzuki/*_tsne/*.png'
'/home/baiqing/PycharmProjects/GraphRXN/result_png/Suzuki'




