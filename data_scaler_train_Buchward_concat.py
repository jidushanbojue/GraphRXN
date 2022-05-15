import os
from glob import glob

base_dir = '/home/baiqing/PycharmProjects/GraphRXN/data_scaler/Buchwald-Hartwig/random_split'
result_dir = '/home/baiqing/PycharmProjects/GraphRXN/result_scaler/Buchward/'
for data_idx in [1,2,3,4,5,6,7,8,9,10]:
    train_path = os.path.join(base_dir, 'FullCV_0' + str(data_idx) + '_train_temp_scaler.csv')
    test_path = os.path.join(base_dir, 'FullCV_0' + str(data_idx) + '_test_temp_scaler.csv')
    result_path = os.path.join(result_dir, 'concat_0'+str(data_idx))
    cmd = f'python reaction_train.py --data_path {train_path} --separate_test_path {test_path}  ' \
          f'--dataset_type regression --num_folds 1 --gpu 0 --epochs 100 --batch_size 128 ' \
          f'--save_dir {result_path} ' \
          f'--metric r2 --reaction_agg_method concat'
    print(cmd)
    os.system(cmd)



#predict
base_dir = '/home/baiqing/PycharmProjects/GraphRXN/data_scaler/Buchwald-Hartwig/random_split'
result_dir = '/home/baiqing/PycharmProjects/GraphRXN/result_scaler/Buchward/'
for data_idx in [1,2,3,4,5,6,7,8,9,10]:
    train_path = os.path.join(base_dir, 'FullCV_0' + str(data_idx) + '_train_temp_scaler.csv')
    test_path = os.path.join(base_dir, 'FullCV_0' + str(data_idx) + '_test_temp_scaler.csv')
    checkpoint_dir = os.path.join(result_dir, 'concat_0'+str(data_idx))
    preds_path = os.path.join(checkpoint_dir, 'preds.csv')
    preds_stat_path = os.path.join(checkpoint_dir, 'stats.csv')
    cmd = f'python predict.py --test_path {test_path}  ' \
          f'--preds_path {preds_path} ' \
          f'--preds_stat_path {preds_stat_path} ' \
          f'--checkpoint_dir {checkpoint_dir}'
    print(cmd)
    os.system(cmd)


#metrics
import os
import pickle
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
result_dir = '/home/baiqing/PycharmProjects/GraphRXN/result_scaler/Buchward/'
scaler_path = '/home/baiqing/PycharmProjects/GraphRXN/scaler_model/Buchwald-Hartwig'
for data_idx in [1,2,3,4,5,6,7,8,9,10]:
    print(data_idx)
    checkpoint_dir = os.path.join(result_dir, 'concat_0' + str(data_idx))
    preds_path = os.path.join(checkpoint_dir, 'preds.csv')
    preds_stat_path = os.path.join(checkpoint_dir, 'stats.csv')
    scaler = pickle.load(open(os.path.join(scaler_path, str(data_idx) + 'pkl'), 'rb'))
    df = pd.read_csv(preds_path)
    df['predict_scaler'] = df['pred_0']
    df['pred_0'] = scaler.inverse_transform(np.array(df['predict_scaler']).reshape(-1,1))
    df.to_csv(preds_path)

    MAE = mean_absolute_error(df['origin_output'], df['pred_0'])
    MSE = mean_squared_error(df['origin_output'], df['pred_0'])
    RMSE = sqrt(MSE)
    r2 = r2_score(df['origin_output'], df['pred_0'])
    with open(preds_stat_path, 'w') as f:
        f.writelines(f'mse,mae,r2_score\n')
        f.writelines(f'{MSE},{MAE},{r2}\n')



df = pd.read_csv(result_path)
RMSE_mean = df['RMSE'].mean()
MAE_mean = df['MAE'].mean()
R2_mean = df['R2'].mean()
idx = len(df)
df.loc[10, :] = ['train_set', 'test_set', RMSE_mean, MAE_mean, R2_mean]
df.to_csv(result_path, index=False)

