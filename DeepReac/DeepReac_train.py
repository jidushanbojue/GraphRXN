import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from dgl.data.utils import split_dataset
from dgllife.utils import CanonicalAtomFeaturizer
from utils import load_dataset, collate_molgraphs_my, EarlyStopping, arg_parse, Rank, run_a_train_epoch_my, run_an_eval_epoch_my, load_data, collate_molgraphs_new
from model_new import DeepReac
import argparse

args = arg_parse()
if args.device == "cpu":
    device = "cpu"
else:
    device = "cuda:"+str(args.device)

# base_dir = '/home/baiqing/PycharmProjects/DeepReac-main_copy/Data/Buchwald-Hartwig/random_split'
# print(base_dir)
#
# result_path = '/result_scaler/Buchwald-Hartwig_result_old.csv'
# if os.path.exists(result_path):
#     os.remove(result_path)
#     print(f'remove {result_path}')

def deepreact_train(train_file, test_file, epochs, stats_file):
    train_set, c_num = load_data(train_file, y_standard=1)
    # val_set = load_data('../Data/random_splits_test/random_split_0_val_temp.csv')
    test_set, c_num = load_data(test_file, y_standard=1)

    train_val_split = [0.8, 0.2]
    train_set, val_set = split_dataset(train_set, frac_list=train_val_split, shuffle=True, random_state=42)

    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_molgraphs_new)
    val_loader = DataLoader(dataset=val_set, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_molgraphs_new)
    test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_molgraphs_new)

    loss_fn = nn.MSELoss(reduction='none')
    in_feats_dim = CanonicalAtomFeaturizer().feat_size('h')
    model = DeepReac(in_feats_dim, len(train_set[0][1]), c_num, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    stopper = EarlyStopping(mode='lower', patience=args.patience)
    model.to(device)

    for epoch in tqdm(range(epochs), total=epochs):
        out_feat_train, index_train, label_train = run_a_train_epoch_my(epoch, model, train_loader, loss_fn, optimizer,
                                                                        args, device)
        val_score, out_feat_val, index_val, label_val, predict_val = run_an_eval_epoch_my(model, val_loader, args,
                                                                                          device)
        early_stop = stopper.step(val_score[0], model)
        if early_stop:
            break
    test_score, out_feat_un, index_un, label_un, predict_un = run_an_eval_epoch_my(model, test_loader, args, device)
    predict_arr = predict_un.data.cpu().numpy()
    test_df = pd.read_csv(test_file)
    test_df['predict'] = predict_arr
    test_df.to_csv(test_file.split('.csv')[0] + '_predict.csv', index=False)
    # label_ratio = len(labeled)/len(data)

    # print("Size of labelled dataset:",100*label_ratio,"%")
    print("Model performance on test dataset: RMSE:", test_score[0], ";MAE:", test_score[1], ";R^2:", test_score[2])
    train_set_name = train_file.rsplit('/', 1)[1]
    test_set_name = test_file.rsplit('/', 1)[1]
    with open(stats_file, 'a') as f:
        f.writelines(f'{train_set_name},{test_set_name},{test_score[0]},{test_score[1]},{test_score[2]}\n')

# for data_idx in [1,2,3,4,5,6,7,8,9,10]:
#     train_path = os.path.join(base_dir, 'FullCV_0'+str(data_idx)+'_train_temp_scaler.csv')
#     test_path = os.path.join(base_dir, 'FullCV_0'+str(data_idx)+'_test_temp_scaler.csv')
#     print(f'train_path: {train_path}')
#     print(f'test_path: {test_path}')
#
#     # data, c_num = load_data(src_file)
#     train_set, c_num = load_data(train_path, y_standard=1)
#     # val_set = load_data('../Data/random_splits_test/random_split_0_val_temp.csv')
#     test_set, c_num = load_data(test_path,y_standard=1)
#
#
#     train_val_split = [0.8, 0.2]
#     train_set, val_set = split_dataset(train_set, frac_list=train_val_split, shuffle=True, random_state=42)
#
#     train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_molgraphs_new)
#     val_loader = DataLoader(dataset=val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_molgraphs_new)
#     test_loader = DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_molgraphs_new)
#
#     loss_fn = nn.MSELoss(reduction='none')
#     in_feats_dim = CanonicalAtomFeaturizer().feat_size('h')
#     model = DeepReac(in_feats_dim, len(train_set[0][1]), c_num, device = device)
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
#     stopper = EarlyStopping(mode='lower', patience=args.patience)
#     model.to(device)
#
#     for epoch in tqdm(range(args.num_epochs), total=args.num_epochs):
#         out_feat_train, index_train, label_train = run_a_train_epoch_my(epoch, model, train_loader, loss_fn, optimizer, args, device)
#         val_score, out_feat_val, index_val, label_val, predict_val = run_an_eval_epoch_my(model, val_loader, args, device)
#         early_stop = stopper.step(val_score[0], model)
#         if early_stop:
#             break
#     test_score, out_feat_un, index_un, label_un, predict_un= run_an_eval_epoch_my(model, test_loader, args, device)
#     predict_arr = predict_un.data.cpu().numpy()
#     test_df = pd.read_csv(test_path)
#     test_df['predict'] = predict_arr
#     test_df.to_csv(test_path.split('.csv')[0]+'_predict.csv', index=False)
#     # label_ratio = len(labeled)/len(data)
#
#     # print("Size of labelled dataset:",100*label_ratio,"%")
#     print("Model performance on test dataset: RMSE:", test_score[0], ";MAE:", test_score[1], ";R^2:", test_score[2])
#     train_set_name = train_path.rsplit('/', 1)[1]
#     test_set_name = test_path.rsplit('/', 1)[1]
#     with open(result_path, 'a') as f:
#         f.writelines(f'{train_set_name},{test_set_name},{test_score[0]},{test_score[1]},{test_score[2]}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reproduce of DeepReac+ code')
    parser.add_argument('-train', '--train_file', type=str, default=None, help='Specify the train file, such as Buchward directory')
    parser.add_argument('-test', '--test_file', type=str, default=None, help='Specify the test file')
    parser.add_argument('-epochs', '--epochs', type=int, default=None, help='Epoch numbers')
    parser.add_argument('-stats', '--stats_file', type=str, default=None, help='Specify the statistics result file')

    train_args = parser.parse_args(['-train', 'data_scaler/Buchward-Hartwig/random_split/FullCV_01_train_temp_scaler.csv',
                                    '-test', 'data_scaler/Buchward-Hartwig/random_split/FullCV_01_test_temp_scaler.csv',
                                    '-epochs', '100',
                                    '-stats', 'result_scaler/Buchward_01_test_stats.csv'])

    # train_args = parser.parse_args()

    deepreact_train(train_args.train_file, train_args.test_file, train_args.epochs, train_args.stats_file)




