from typing import List

import torch
import torch.nn as nn
from tqdm import trange
import pandas as pd

from chemprop.data.data import ReactionDataset, StandardScaler


def predict(model: nn.Module,
            data: ReactionDataset,
            batch_size: int,
            scaler: StandardScaler = None) -> List[List[float]]:
    """
    Makes predictions on a dataset using an ensemble of models.

    :param model: A model.
    :param data: A MoleculeDataset.
    :param batch_size: Batch size.
    :param scaler: A StandardScaler object fit on the training targets.
    :return: A list of lists of predictions. The outer list is examples
    while the inner list is tasks.
    """
    model.eval()

    preds = []
    encoder_output_list = []

    num_iters, iter_step = len(data), batch_size


    for i in range(0, num_iters, iter_step):
        # Prepare batch
        mol_batch = ReactionDataset(data[i:i + batch_size])
        smiles_batch, features_batch = mol_batch.smiles(), mol_batch.features()
        temp_batch = mol_batch.temps()
        temp_batch = torch.Tensor([[0 if x is None else x for x in tb] for tb in temp_batch])

        target_batch = mol_batch.targets()
        targets = torch.Tensor([[0 if x is None else x for x in tb] for tb in target_batch])
        # Run model
        batch = smiles_batch

        with torch.no_grad():
            encoder_output, batch_preds = model(batch, features_batch, temp_batch)
            encoder_output = encoder_output.to('cpu')
        encoder_output = torch.cat([encoder_output, targets], axis=1)
        batch_preds = batch_preds.data.cpu().numpy()

        # Inverse scale if regression
        if scaler is not None:
            batch_preds = scaler.inverse_transform(batch_preds)

        # Collect vectors
        batch_preds = batch_preds.tolist()
        preds.extend(batch_preds)

        encoder_output = encoder_output.tolist()

        encoder_output_list.extend(encoder_output)

    encoder_df = pd.DataFrame(encoder_output_list)

        # encoder_tensor = torch.cat(encoder_output_list)


    return encoder_df, preds
