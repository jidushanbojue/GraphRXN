import typer
import torch
import pkg_resources
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
from rxnfp.models import SmilesClassificationModel
import sklearn
from rdkit import Chem
import pandas as pd

try:
    import wandb
    wandb_available = True
except ImportError:
    wandb_available = False

app = typer.Typer()


def canonicalize_with_dict(smi, can_smi_dict={}):
    if smi not in can_smi_dict.keys():
        return Chem.MolToSmiles(Chem.MolFromSmiles(smi))
    else:
        return can_smi_dict[smi]

def generate_inhouse_rxns(df):
    df = df.copy()
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

#
# NAME_SPLIT = [
#     ('all', 'fold1'), ('all', 'fold2'), ('all', 'fold3'), ('all', 'fold4'),('all', 'fold5'),
#     ('G1_1909-25-01_317', 'fold1'), ('G1_1909-25-01_317', 'fold2'), ('G1_1909-25-01_317', 'fold3'), ('G1_1909-25-01_317', 'fold4'),('G1_1909-25-01_317', 'fold5'),
#     ('G2_1909-26-01_419', 'fold1'), ('G2_1909-26-01_419', 'fold2'), ('G2_1909-26-01_419', 'fold3'), ('G2_1909-26-01_419', 'fold4'),('G2_1909-26-01_419', 'fold5'),
#     ('G3_1904-25-01_401', 'fold1'), ('G3_1904-25-01_401', 'fold2'), ('G3_1904-25-01_401', 'fold3'), ('G3_1904-25-01_401', 'fold4'),('G3_1904-25-01_401', 'fold5'),
#     ('G4_1909-27-01_421', 'fold1'), ('G4_1909-27-01_421', 'fold2'), ('G4_1909-27-01_421', 'fold3'), ('G4_1909-27-01_421', 'fold4'),('G4_1909-27-01_421', 'fold5')
# ]

# NAME_SPLIT = [
#     ('G1_1909-25-01_317', 'fold1'), ('G1_1909-25-01_317', 'fold2'), ('G1_1909-25-01_317', 'fold3'), ('G1_1909-25-01_317', 'fold4'),('G1_1909-25-01_317', 'fold5'),
#     ('G2_1909-26-01_419', 'fold1'), ('G2_1909-26-01_419', 'fold2'), ('G2_1909-26-01_419', 'fold3'), ('G2_1909-26-01_419', 'fold4'),('G2_1909-26-01_419', 'fold5'),
#     ('G3_1904-25-01_401', 'fold1'), ('G3_1904-25-01_401', 'fold2'), ('G3_1904-25-01_401', 'fold3'), ('G3_1904-25-01_401', 'fold4'),('G3_1904-25-01_401', 'fold5'),
#     ('G4_1909-27-01_421', 'fold1'), ('G4_1909-27-01_421', 'fold2'), ('G4_1909-27-01_421', 'fold3'), ('G4_1909-27-01_421', 'fold4'),('G4_1909-27-01_421', 'fold5')
# ]

def launch_training_on_all_splits(experiment: str, splits, base_model: str, dropout: float, learning_rate: float):
    project = f'denmark_03_dataset3_training_{experiment}_{base_model}'

    model_args = {
        'num_train_epochs': 50, 'overwrite_output_dir': True,
        'learning_rate': learning_rate, 'gradient_accumulation_steps': 1,
        'regression': True, "num_labels": 1, "fp16": False,
        "evaluate_during_training": True, 'manual_seed': 42,
        "max_seq_length": 300, "train_batch_size": 16, "warmup_ratio": 0.00,
        "config": {'hidden_dropout_prob': dropout}}

    for folder,name in splits:
        if wandb_available: wandb.init(name=name, project=project, reinit=True)

        train_df = pd.read_excel('../data/Dataset_inhouse/' + folder+'/'+ name + '_train.xlsx')
        test_df = pd.read_excel('../data/Dataset_inhouse/'  + folder+'/'+ name + '_test.xlsx')

        train_df['text'] = generate_inhouse_rxns(train_df)
        test_df['text'] = generate_inhouse_rxns(test_df)

        train_df['labels'] = train_df['Output']
        test_df['labels'] = test_df['Output']

        mean = train_df.labels.mean()
        std = train_df.labels.std()
        train_df['labels'] = (train_df['labels'] - mean) / std
        test_df['labels'] = (test_df['labels'] - mean) / std

        model_path = pkg_resources.resource_filename("rxnfp", f"models/transformers/bert_{base_model}")
        pretrained_bert = SmilesClassificationModel("bert", model_path, num_labels=1, args=model_args, use_cuda=torch.cuda.is_available())
        pretrained_bert.train_model(train_df, output_dir=f"{folder}/outputs_0415_all_inhouse_{experiment}_{base_model}_{name}_split", eval_df=test_df, r2=sklearn.metrics.r2_score)
        if wandb_available: wandb.join() # multiple runs in same script


@app.command()
def pretrained():
    """
    Launch training and evaluation with a pretrained base model from rxnfp.
    """
    launch_training_on_all_splits(experiment='full', splits=NAME_SPLIT, base_model='pretrained', dropout=0.7987, learning_rate=0.00009659)



if __name__ == '__main__':
    app()
