# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 19:42:36 2019

@author: SY
"""
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from chemprop.parsing import parse_predict_args, modify_predict_args
from chemprop.train import make_predictions


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def get_metrics(y_true, y_pred):
    return {'mse': mean_squared_error(y_true=y_true, y_pred=y_pred),
            'mae': mean_absolute_error(y_true=y_true, y_pred=y_pred),
            'r2_score': r2_score(y_true=y_true, y_pred=y_pred)}



if __name__ == '__main__':
    args = parse_predict_args()
    # args.checkpoint_dir = './ckpt'
    modify_predict_args(args)
    
    df = pd.read_csv(args.test_path)
    df.rename(columns={df.columns[2]: 'Output'}, inplace=True)
    pred, smiles = make_predictions(args)
    # df = pd.DataFrame({'smiles':smiles})

    for i in range(len(pred[0])):
        df[f'pred_{i}'] = [item[i] for item in pred]
        all_metrics = get_metrics(df['Output'], df[f'pred_{i}'])
    # df.to_csv(f'./predict.csv', index=False)
    df_stats = pd.DataFrame(all_metrics, index=[0])
    df_stats.to_csv((args.preds_stat_path))

    df.to_csv(args.preds_path, index=False)
