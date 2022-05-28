import typer
import torch
import pkg_resources
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import pandas as pd
from typing import List, Tuple

from rxnfp.models import SmilesClassificationModel
#from rxn_yields.data import generate_buchwald_hartwig_rxns
import sklearn

try:
    import wandb
    wandb_available = True
except ImportError:
    wandb_available = False

app = typer.Typer()

NAME_SPLIT = [
    'FullCV_01', 'FullCV_02', 'FullCV_03', 'FullCV_04',
    'FullCV_05', 'FullCV_06','FullCV_07','FullCV_08',
    'FullCV_09', 'FullCV_10'
]


def launch_training_on_all_splits(experiment: str, splits, base_model: str, dropout: float, learning_rate: float):
    project = f'denmark_03_dataset3_training_{experiment}_{base_model}'

    model_args = {
        'num_train_epochs': 50, 'overwrite_output_dir': True,
        'learning_rate': learning_rate, 'gradient_accumulation_steps': 1,
        'regression': True, "num_labels": 1, "fp16": False,
        "evaluate_during_training": True, 'manual_seed': 42,
        "max_seq_length": 300, "train_batch_size": 16, "warmup_ratio": 0.00,
        "config": {'hidden_dropout_prob': dropout}}

    for name in splits:
        if wandb_available: wandb.init(name=name, project=project, reinit=True)

        train_df = pd.read_csv('../data/Denmark/'+ name +'_train_products_generated.csv')
        test_df = pd.read_csv('../data/Denmark/'+ name +'_test_products_generated.csv')
        train_df = train_df[['bert_rxns', 'Output']]
        test_df = test_df[['bert_rxns', 'Output']]

        train_df.columns = ['text', 'labels']
        test_df.columns = ['text', 'labels']
        mean = train_df.labels.mean()
        std = train_df.labels.std()
        train_df['labels'] = (train_df['labels'] - mean) / std
        test_df['labels'] = (test_df['labels'] - mean) / std

        model_path = pkg_resources.resource_filename("rxnfp", f"models/transformers/bert_{base_model}")
        pretrained_bert = SmilesClassificationModel("bert", model_path, num_labels=1, args=model_args, use_cuda=torch.cuda.is_available())
        pretrained_bert.train_model(train_df, output_dir=f"outputs_0421_10CV_denmark_{experiment}_{base_model}_{name}_split", eval_df=test_df, r2=sklearn.metrics.r2_score)
        if wandb_available: wandb.join() # multiple runs in same script


@app.command()
def pretrained():
    """
    Launch training and evaluation with a pretrained base model from rxnfp.
    """
    launch_training_on_all_splits(experiment='full', splits=NAME_SPLIT, base_model='pretrained', dropout=0.7987, learning_rate=0.00009659)



if __name__ == '__main__':
    app()
