import logging
import sklearn
import torch
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

torch.cuda.is_available()

def make_plot(y_test, y_pred, rmse, r2_score, mae, name):
    fontsize = 16
    fig, ax = plt.subplots(figsize=(8, 8))
    # r2_patch = mpatches.Patch(label="R2 = {:.3f}".format(r2_score), color="#5402A3")
    # rmse_patch = mpatches.Patch(label="RMSE = {:.3f}".format(rmse), color="#5402A3")
    # mae_patch = mpatches.Patch(label="MAE = {:.3f}".format(mae), color="#5402A3")

    r2_patch = mpatches.Patch(label="R2 = {:.3f}".format(r2_score), color="darkcyan")
    rmse_patch = mpatches.Patch(label="RMSE = {:.3f}".format(rmse), color="darkcyan")
    mae_patch = mpatches.Patch(label="MAE = {:.3f}".format(mae), color="darkcyan")

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    # plt.scatter(y_pred, y_test, alpha=0.2, color="#5402A3")
    plt.scatter(y_test, y_pred, alpha=0.1, color="darkcyan")
    plt.plot(np.arange(100), np.arange(100), ls="--", c=".3")
    plt.legend(handles=[r2_patch, rmse_patch, mae_patch], fontsize=fontsize)
    ax.set_ylabel('Predicted', fontsize=fontsize)
    ax.set_xlabel('Observerd', fontsize=fontsize)
    ax.set_title(name, fontsize=fontsize)
    plt.savefig(name+'.png', dpi=300)
    plt.show()
    return fig


# %%
# data
y_predictions = []
y_tests = []
r2_scores = []
rmse_scores = []





if __name__ == '__main__':
    base_dir = '/data/baiqing/PycharmProjects/GraphRXN/stat'
    concat_file = os.path.join(base_dir, 'in_house_concat_preds_ensemble.csv')
    concat_info = {'r2_score': 0.713,
                   'MAE': 0.063,
                   'RMSE': 0.087,
                   'name': 'GraphRXN-concat'}

    sum_file = os.path.join(base_dir, 'in_house_sum_preds_ensemble.csv')
    sum_info = {'r2_score': 0.704,
                'MAE': 0.064,
                'RMSE': 0.089,
                'name': 'GraphRXN-sum'}



    df = pd.read_csv(concat_file)
    make_plot(df['True'], df['predict'], concat_info['RMSE'], concat_info['r2_score'], concat_info['MAE'], concat_info['name'])

    df = pd.read_csv(sum_file)
    make_plot(df['True'], df['predict'], sum_info['RMSE'], sum_info['r2_score'], sum_info['MAE'], sum_info['name'])



