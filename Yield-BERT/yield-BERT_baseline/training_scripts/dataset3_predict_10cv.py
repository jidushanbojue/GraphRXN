import torch
from rxnfp.models import SmilesClassificationModel
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# data

NAME_SPLIT = [
    'FullCV_01', 'FullCV_02', 'FullCV_03', 'FullCV_04',
    'FullCV_05', 'FullCV_06','FullCV_07','FullCV_08',
    'FullCV_09', 'FullCV_10'
]

# data
r2_list =[]
rmse_list=[]
mae_list=[]

for name in NAME_SPLIT:
    train_df = pd.read_csv('../data/Denmark/' + name + '_train_products_generated.csv')
    test_df = pd.read_csv('../data/Denmark/' + name + '_test_products_generated.csv')

    train_df = train_df[['bert_rxns', 'Output']]
    test_df = test_df[['bert_rxns', 'Output']]

    train_df.columns = ['text', 'labels']
    test_df.columns = ['text', 'labels']
    mean = train_df.labels.mean()
    std = train_df.labels.std()
    train_df['labels'] = (train_df['labels'] - mean) / std
    test_df['labels'] = (test_df['labels'] - mean) / std
    train_df.head()
    ## Predictions

    model_path = 'outputs_0421_10CV_denmark_full_pretrained_'+ name +'_split/checkpoint-570-epoch-15'
    model= SmilesClassificationModel('bert', model_path,
                                     num_labels=1, args={"regression": True},
                                     use_cuda=torch.cuda.is_available())

    # yield_predicted = trained_yield_bert.predict(test_df.head(10).text.values)[0]
    # yield_predicted = yield_predicted * std + mean

    y_test = test_df.labels.values
    y_test = y_test * std + mean

    y_preds = model.predict(test_df.text.values)[0]
    y_preds = y_preds * std + mean
    #y_preds = np.clip(y_preds, 0, 100)

    r_squared = r2_score(y_test, y_preds)
    rmse = mean_squared_error(y_test, y_preds) ** 0.5
    mae = mean_absolute_error(y_test, y_preds)

    print(name +" result:")
    print("R2=" + str(r_squared))
    print("RMSE=" + str(rmse))
    print("MAE=" + str(mae))
    print()

    r2_list.append(r_squared)
    rmse_list.append(rmse)
    mae_list.append(mae)



data = {
        'r2':r2_list,
        'rmse':rmse_list,
        'mae':mae_list
}

df = pd.DataFrame(data,index=NAME_SPLIT)
df.to_excel("Denmark_0421_pretrained_bert_10CV.xlsx")
