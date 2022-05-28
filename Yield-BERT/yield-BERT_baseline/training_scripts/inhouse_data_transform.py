
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

df = pd.read_excel('../data/Dataset_inhouse/all/fold1_test.xlsx')

#converted_rxns = generate_inhouse_rxns(df)
# print(converted_rxns)

df['rxn'] = generate_inhouse_rxns(df)
train_df = df[['rxn', 'Output']]

print(train_df)
