import torch
from rxnfp.models import SmilesClassificationModel
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# data
from rdkit import Chem
import pandas as pd


def canonicalize_with_dict(smi, can_smi_dict={}):
    if smi not in can_smi_dict.keys():
        return Chem.MolToSmiles(Chem.MolFromSmiles(smi))
    else:
        return can_smi_dict[smi]

def generate_inhouse_rxns(df):
    df = df.copy()
    #products = df['Product']
    rxns = []
    can_smiles_dict = {}
    for i, row in df.iterrows():
        amine = canonicalize_with_dict(row['Amine'], can_smiles_dict)
        can_smiles_dict[row['Amine']] = amine
        bromide = canonicalize_with_dict(row['Bromide'], can_smiles_dict)
        can_smiles_dict[row['Bromide']] = bromide
        product = canonicalize_with_dict(row['Product'], can_smiles_dict)
        can_smiles_dict[row['Product']] = product

        reactants = f"{amine}.{bromide}"
        rxns.append(f"{reactants}>>{product}")
    return rxns

NAME_SPLIT = [
    ('all', 'fold1','3900'), ('all', 'fold2','3900'), ('all', 'fold3','3900'), ('all', 'fold4','3900'),('all', 'fold5','3900'),
    ('G1_1909-25-01_317', 'fold1','800'), ('G1_1909-25-01_317', 'fold2','800'), ('G1_1909-25-01_317', 'fold3','800'), ('G1_1909-25-01_317', 'fold4','800'),('G1_1909-25-01_317', 'fold5','800'),
    ('G2_1909-26-01_419', 'fold1','1050'), ('G2_1909-26-01_419', 'fold2','1050'), ('G2_1909-26-01_419', 'fold3','1050'), ('G2_1909-26-01_419', 'fold4','1050'),('G2_1909-26-01_419', 'fold5','1050'),
    ('G3_1904-25-01_401', 'fold1','1000'), ('G3_1904-25-01_401', 'fold2','1050'), ('G3_1904-25-01_401', 'fold3','1050'), ('G3_1904-25-01_401', 'fold4','1050'),('G3_1904-25-01_401', 'fold5','1050'),
    ('G4_1909-27-01_421', 'fold1','1050'), ('G4_1909-27-01_421', 'fold2','1100'), ('G4_1909-27-01_421', 'fold3','1100'), ('G4_1909-27-01_421', 'fold4','1100'),('G4_1909-27-01_421', 'fold5','1100')
]

# data
r2_list =[]
rmse_list=[]
mae_list=[]
namelist=[]

for folder,name,size in NAME_SPLIT:
    train_df = pd.read_excel('../data/Dataset_inhouse/' + folder + '/' + name + '_train.xlsx')
    test_df = pd.read_excel('../data/Dataset_inhouse/' + folder + '/' + name + '_test.xlsx')

    train_df['text'] = generate_inhouse_rxns(train_df)
    test_df['text'] = generate_inhouse_rxns(test_df)

    train_df['labels'] = train_df['Output']
    test_df['labels'] = test_df['Output']

    mean = train_df.labels.mean()
    std = train_df.labels.std()
    train_df['labels'] = (train_df['labels'] - mean) / std
    test_df['labels'] = (test_df['labels'] - mean) / std
    train_df.head()
    ## Predictions

    model_path = f"{folder}/outputs_0415_all_inhouse_full_pretrained_{name}_split/checkpoint-{size}-epoch-50"
    model= SmilesClassificationModel('bert', model_path,
                                     num_labels=1, args={"regression": True},
                                     use_cuda=torch.cuda.is_available())

    # yield_predicted = trained_yield_bert.predict(test_df.head(10).text.values)[0]
    # yield_predicted = yield_predicted * std + mean

    y_test = test_df.labels.values
    y_test = y_test * std + mean

    y_preds = model.predict(test_df.text.values)[0]
    y_preds = y_preds * std + mean
    y_preds = np.clip(y_preds, 0, 1)

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
    namelist.append(folder+name)


data = {
        'r2':r2_list,
        'rmse':rmse_list,
        'mae':mae_list,
}

df = pd.DataFrame(data,index=namelist)
df.to_excel("inhouse_bert_5CV_5dataset.xlsx")
