import torch
import pkg_resources
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import pandas as pd
from typing import List, Tuple

from rxn_yields.core import SmilesClassificationModel
import sklearn

try:
    import wandb
    wandb_available = True
except ImportError:
    raise ValueError('Wandb is not available')


def main():
    def train():
        # import os
        #
        # os.environ["WANDB_API_KEY"] = "b9650e2d829f9c2f179ac29f0703222d8e9cafdb"
        # os.environ["WANDB_MODE"] = "offline"
        # import os
        #
        # os.environ["WANDB_API_KEY"] = "b9650e2d829f9c2f179ac29f0703222d8e9cafdb"
        # os.environ["WANDB_MODE"] = "offline"
        #
        wandb.init(mode="disabled")
        #wandb.init()
        print("HyperParams=>>", wandb.config)
        model_args = {
            'wandb_project': "denmark_random_01",
            'num_train_epochs': 50, 'overwrite_output_dir': True,
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
        pretrained_bert.train_model(train_df, output_dir='../trained_models/denmark/outputs_hyperparam_pretrained_bert_dataset3_CV_01',
                                    eval_df=test_df, r2=sklearn.metrics.r2_score)

    train_df = pd.read_csv('../data/Denmark/FullCV_01_train_products_generated.csv')
    test_df = pd.read_csv('../data/Denmark/FullCV_01_test_products_generated.csv')

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


    sweep_id = wandb.sweep(sweep_config, project="denmark_random_02_hyperparams_sweep")
    wandb.agent(sweep_id, function=train)


if __name__ == '__main__':
    main()