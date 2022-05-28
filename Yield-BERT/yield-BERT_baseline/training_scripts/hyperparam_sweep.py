import torch
import pkg_resources
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import pandas as pd
from rxn_yields.core import SmilesClassificationModel
import sklearn

from rdkit import Chem
from rdkit.Chem import rdChemReactions
try:
    import wandb
    wandb_available = True
except ImportError:
    raise ValueError('Wandb is not available')

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


def main():
    def train():
        wandb.init(mode="disabled")
        #wandb.init()
        print("HyperParams=>>", wandb.config)
        model_args = {
            'wandb_project': "inhouse_hyperparam_sweep",
            'num_train_epochs': 10, 'overwrite_output_dir': True,
            'gradient_accumulation_steps': 1, "warmup_ratio": 0.00,
            "train_batch_size": 16, 'regression': True, "num_labels": 1,
            "fp16": False, "evaluate_during_training": True,
            "max_seq_length": 300,
            "config": {
                'hidden_dropout_prob': wandb.config.dropout_rate,
            },
            'learning_rate': wandb.config.learning_rate,
        }
        model_path = pkg_resources.resource_filename("rxnfp", f"models/transformers/bert_pretrained")
        pretrained_bert = SmilesClassificationModel("bert", model_path, num_labels=1, args=model_args,
                                                    use_cuda=torch.cuda.is_available())
        pretrained_bert.train_model(train_df, output_dir='../Dataset_inhouse/log_file/outputs_inhouse_hyperparam_sweeep_01',
                                    eval_df=test_df, r2=sklearn.metrics.r2_score)

    train_df = pd.read_excel('../data/Dataset_inhouse/all/fold1_train_2022-04-07_shuffle-fix_final_rm16_1558.xlsx')
    test_df = pd.read_excel('../data_products_generated/Dataset_inhouse/all/fold1_test.xlsx')


    train_df = train_df[['bert_rxns', 'Output']]
    test_df = test_df[['bert_rxns', 'Output']]

    train_df.columns = ['text', 'labels']
    test_df.columns = ['text', 'labels']

    mean = train_df.labels.mean()
    std = train_df.labels.std()

    train_df['labels'] = (train_df['labels'] - mean) / std
    test_df['labels'] = (test_df['labels'] - mean) / std

    sweep_config = {
        'method': 'bayes',  # grid, random, bayes
        'metric': {
            'name': 'r2',
            'goal': 'maximize'
        },
        'parameters': {
            'learning_rate': {
                'min': 1e-6,
                'max': 1e-4

            },
            'dropout_rate': {
                'min': 0.05,
                'max': 0.8
            }
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="inhouse_random_01_hyperparams_sweep")
    wandb.agent(sweep_id, function=train)


if __name__ == '__main__':
    main()