import os
from glob import glob
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler


base_dir = '/home/baiqing/PycharmProjects/DeepReac-main_copy/Data/Buchwald-Hartwig/random_split'
scaler_path = '/home/baiqing/PycharmProjects/DeepReac-main_copy/Data/scaler_model/Buchwald-Hartwig'
for data_idx in [1,2,3,4,5,6,7,8,9,10]:
    train_path = os.path.join(base_dir, 'FullCV_0'+str(data_idx)+'_train_temp.csv')
    test_path = os.path.join(base_dir, 'FullCV_0'+str(data_idx)+'_test_temp.csv')
    train_scaler_path = os.path.join(base_dir, 'FullCV_0'+str(data_idx)+'_train_temp_scaler.csv')
    test_scaler_path = os.path.join(base_dir, 'FullCV_0'+str(data_idx)+'_test_temp_scaler.csv')
    print(train_path)
    print(test_path)

    train_set = pd.read_csv(train_path)
    test_set = pd.read_csv(test_path)
    data_all = pd.concat([train_set,test_set], axis=0)
    data_all['origin_output'] = data_all['output']
    scaler = StandardScaler()
    output = scaler.fit_transform(np.array(data_all['output']).reshape(-1,1))
    data_all['output'] = output
    data_all[:len(train_set)].to_csv(train_scaler_path, index=False)
    data_all[len(train_set):].to_csv(test_scaler_path, index=False)

    pickle.dump(scaler, open(os.path.join(scaler_path, str(data_idx)+'pkl'), 'wb'))
    # scaler = pickle.load(open(os.path.join(scaler_path, str(i)+'pkl'),'rb'))
    # origin_data = scaler.inverse_transform(output)




base_dir = '/home/baiqing/PycharmProjects/DeepReac-main_copy/Data/random_splits_test'
scaler_path = '/home/baiqing/PycharmProjects/DeepReac-main_copy/Data/scaler_model/random_splits_test'
for data_idx in [0,1,2,3,4,5,6,7,8,9]:
    train_path = os.path.join(base_dir, 'random_split_'+str(data_idx)+'_train_temp.csv')
    test_path = os.path.join(base_dir, 'random_split_'+str(data_idx)+'_test_temp.csv')
    train_scaler_path = os.path.join(base_dir, 'random_split_'+str(data_idx)+'_train_temp_scaler.csv')
    test_scaler_path = os.path.join(base_dir, 'random_split_'+str(data_idx)+'_test_temp_scaler.csv')
    print(train_path)
    print(test_path)

    train_set = pd.read_csv(train_path)
    test_set = pd.read_csv(test_path)
    data_all = pd.concat([train_set, test_set], axis=0)
    data_all['origin_output'] = data_all['output']
    scaler = StandardScaler()
    output = scaler.fit_transform(np.array(data_all['output']).reshape(-1, 1))
    data_all['output'] = output
    data_all[:len(train_set)].to_csv(train_scaler_path, index=False)
    data_all[len(train_set):].to_csv(test_scaler_path, index=False)

    pickle.dump(scaler, open(os.path.join(scaler_path, str(data_idx) + 'pkl'), 'wb'))
    # scaler = pickle.load(open(os.path.join(scaler_path, str(i)+'pkl'),'rb'))
    # origin_data = scaler.inverse_transform(output)






