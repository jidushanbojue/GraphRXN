import logging
import sklearn
import torch
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import math

# torch.cuda.is_available()

def make_plot(y_test, y_pred, rmse, r2_score, mae, name, save_path):
    fontsize = 16
    fig, ax = plt.subplots(figsize=(8, 8))

    r2_patch = mpatches.Patch(label="R2 = {:.3f}".format(r2_score), color="darkcyan")
    rmse_patch = mpatches.Patch(label="RMSE = {:.3f}".format(rmse), color="darkcyan")
    mae_patch = mpatches.Patch(label="MAE = {:.3f}".format(mae), color="darkcyan")
    ##Denmark##
    # plt.xlim(min(y_test)-0.1, max(y_test)+0.1)
    # plt.ylim(min(y_pred)-0.1, max(y_pred)+0.1)
    ##Suzuki##
    # plt.xlim(-0.05, 1.05)
    # plt.ylim(-0.05, 1.05)
    ####Buchwald or Suzuki*100####
    plt.xlim(-5, 105)
    plt.ylim(-5, 105)
    # plt.scatter(y_pred, y_test, alpha=0.2, color="#5402A3")
    plt.scatter(y_test, y_pred, alpha=0.1, color="darkcyan")
    plt.plot(np.arange(100), np.arange(100), ls="--", c=".3")
    plt.legend(handles=[r2_patch, rmse_patch, mae_patch], fontsize=fontsize)
    ax.set_ylabel('Predicted', fontsize=fontsize)
    ax.set_xlabel('Observerd', fontsize=fontsize)
    ax.set_title(name, fontsize=fontsize)
    plt.savefig(save_path, dpi=300)
    plt.show()
    return fig
if __name__ == '__main__':
    # for func in ['concat', 'sum']:
    #     fnames = glob(f'/home/baiqing/PycharmProjects/GraphRXN/result_scaler/Suzuki/{func}_*')
    #     for fname in fnames:
    #         # os.remove(os.path.join(fname,'plot_scatter.png'))
    #         # i = fname.rsplit('/', 1)[1].split('_')[1]
    #
    #         i = str(int(fname.rsplit('/', 1)[1].split('_')[1])+1)
    #         if len(i)==1:
    #             i = '0'+i
    #
    #         preds = os.path.join(fname, 'preds.csv')
    #         stats = os.path.join(fname, 'stats.csv')
    #         preds_df = pd.read_csv(preds)
    #         stats_df = pd.read_csv(stats)
    #         info = {
    #             'name': f'Suzuki GraphRXN-{func} of fold ' + i,
    #             'RMSE': math.sqrt(stats_df.loc[0,'mse']),
    #             'MAE': stats_df.loc[0,'mae'],
    #             'r2_score': stats_df.loc[0,'r2_score']
    #                 }
    #         ##Suzuki##
    #         preds_df['pred_0'] = np.clip(preds_df['pred_0'], 0, 1)
    #         ##Buchwald##
    #         # preds_df['pred_0'] = np.clip(preds_df['pred_0'], 0, 100)
    #         save_path = os.path.join(fname, info['name'].replace(' ', '_')+'.png')
    #         make_plot(preds_df['origin_output'], preds_df['pred_0'], info['RMSE'], info['r2_score'], info['MAE'], info['name'], save_path)


    #Suzuki*100
    for func in ['concat', 'sum']:
        fnames = glob(f'/home/baiqing/PycharmProjects/GraphRXN/result_scaler/Suzuki/{func}_*')
        for fname in fnames:
            # os.remove(os.path.join(fname,'plot_scatter.png'))
            # i = fname.rsplit('/', 1)[1].split('_')[1]

            i = str(int(fname.rsplit('/', 1)[1].split('_')[1])+1)
            if len(i)==1:
                i = '0'+i

            preds = os.path.join(fname, 'preds.csv')
            stats = os.path.join(fname, 'stats_100.csv')
            preds_df = pd.read_csv(preds)
            stats_df = pd.read_csv(stats)
            info = {
                'name': f'Suzuki GraphRXN-{func} of fold ' + i,
                'RMSE': math.sqrt(stats_df.loc[0,'mse']),
                'MAE': stats_df.loc[0,'mae'],
                'r2_score': stats_df.loc[0,'r2_score']
                    }
            ##Suzuki##
            # preds_df['pred_0'] = np.clip(preds_df['pred_0'], 0, 1)
            ##Suzuki*100##
            preds_df['pred_0'] = np.clip(preds_df['pred_0']*100, 0, 100)
            save_path = os.path.join(fname, info['name'].replace(' ', '_')+'_100.png')
            make_plot(preds_df['origin_output']*100, preds_df['pred_0'], info['RMSE'], info['r2_score'], info['MAE'], info['name'], save_path)





