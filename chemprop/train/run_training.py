from argparse import Namespace
import csv
from logging import Logger
import os
# from pprint import pformat
from typing import List

import numpy as np
from tensorboardX import SummaryWriter
import torch
import pickle
from torch.optim.lr_scheduler import ExponentialLR

from .evaluate import evaluate, evaluate_predictions
from .predict import predict
from .train import train
from chemprop.data import StandardScaler
from chemprop.data.utils import get_class_sizes, get_data, get_task_names, split_data
from chemprop.models.model import build_model
from chemprop.nn_utils import param_count
from chemprop.utils import build_optimizer, build_lr_scheduler, get_loss_func, get_metric_func, load_checkpoint, \
    makedirs, save_checkpoint
import pandas as pd


def get_data_df(moleculeDataSet):
    df = pd.DataFrame()
    smiles_list = []
    targets_list = []
    for datapoint in moleculeDataSet.data:
        smiles_list.append(datapoint.smiles)
        targets_list.append(datapoint.targets[0])
    df['smiles'] = smiles_list
    df['targets'] = targets_list
    return df


def run_training(args: Namespace, logger: Logger = None) -> List[float]:
    """
    Trains a model and returns test scores on the model checkpoint with the highest validation score.

    :param args: Arguments.
    :param logger: Logger.
    :return: A list of ensemble scores for each task.
    """
    if logger is not None:
        debug, info = logger.debug, logger.info
    else:
        debug = info = print

    # Set GPU
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    # Print args
    # =============================================================================
    #     debug(pformat(vars(args)))
    # =============================================================================

    # Get data
    debug('Loading data')
    args.task_names = get_task_names(args.data_path)
    data = get_data(path=args.data_path, args=args, logger=logger)
    args.num_tasks = data.num_tasks()
    args.features_size = data.features_size()
    debug(f'Number of tasks = {args.num_tasks}')

    # Split data
    debug(f'Splitting data with seed {args.seed}')
    if args.separate_test_path:
        test_data = get_data(path=args.separate_test_path, args=args, features_path=args.separate_test_features_path,
                             logger=logger)
    if args.separate_val_path:
        val_data = get_data(path=args.separate_val_path, args=args, features_path=args.separate_val_features_path,
                            logger=logger)

    if args.separate_val_path and args.separate_test_path:
        train_data = data
    elif args.separate_val_path:
        train_data, _, test_data = split_data(data=data, split_type=args.split_type, sizes=(0.8, 0.2, 0.0),
                                              seed=args.seed, args=args, logger=logger)
    elif args.separate_test_path:
        train_data, val_data, _ = split_data(data=data, split_type=args.split_type, sizes=(0.8, 0.2, 0.0),
                                             seed=args.seed, args=args, logger=logger)
    else:
        print('=' * 100)
        train_data, val_data, test_data = split_data(data=data, split_type=args.split_type, sizes=args.split_sizes,
                                                     seed=args.seed, args=args, logger=logger)
        ###my_code###
        # train_df = get_data_df(train_data)
        # train_df.to_csv('~/PycharmProjects/CMPNN-master/data/24w_train_df_seed0.csv')
        # val_df = get_data_df(val_data)
        # val_df.to_csv('~/PycharmProjects/CMPNN-master/data/24w_val_df_seed0.csv')
        # test_df = get_data_df(test_data)
        # test_df.to_csv('~/PycharmProjects/CMPNN-master/data/24w_test_df_seed0.csv')

        ##########

    if args.dataset_type == 'classification':
        class_sizes = get_class_sizes(data)
        debug('Class sizes')
        for i, task_class_sizes in enumerate(class_sizes):
            debug(f'{args.task_names[i]} '
                  f'{", ".join(f"{cls}: {size * 100:.2f}%" for cls, size in enumerate(task_class_sizes))}')

    if args.save_smiles_splits:
        with open(args.data_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)

            lines_by_smiles = {}
            indices_by_smiles = {}
            for i, line in enumerate(reader):
                smiles = line[0]
                lines_by_smiles[smiles] = line
                indices_by_smiles[smiles] = i

        all_split_indices = []
        for dataset, name in [(train_data, 'train'), (val_data, 'val'), (test_data, 'test')]:
            with open(os.path.join(args.save_dir, name + '_smiles.csv'), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['smiles'])
                for smiles in dataset.smiles():
                    # writer.writerow([smiles])
                    # writer.writerows(smiles)
                    print('*'.join(smiles))
                    # writer.writerow('abc')
                    writer.writerow(['*'.join(smiles)])
            with open(os.path.join(args.save_dir, name + '_full.csv'), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for smiles in dataset.smiles():
                    # writer.writerow(lines_by_smiles[smiles])
                    writer.writerow(lines_by_smiles['*'.join(smiles)])
            split_indices = []
            for smiles in dataset.smiles():
                # split_indices.append(indices_by_smiles[smiles])
                split_indices.append(indices_by_smiles['*'.join(smiles)])
                split_indices = sorted(split_indices)
            all_split_indices.append(split_indices)
        with open(os.path.join(args.save_dir, 'split_indices.pckl'), 'wb') as f:
            pickle.dump(all_split_indices, f)

    if args.features_scaling:
        features_scaler = train_data.normalize_features(replace_nan_token=0)
        val_data.normalize_features(features_scaler)
        test_data.normalize_features(features_scaler)
    else:
        features_scaler = None

    args.train_data_size = len(train_data)

    debug(f'Total size = {len(data):,} | '
          f'train size = {len(train_data):,} | val size = {len(val_data):,} | test size = {len(test_data):,}')

    # Initialize scaler and scale training targets by subtracting mean and dividing standard deviation (regression only)
    if args.dataset_type == 'regression':
        debug('Fitting scaler')
        train_smiles, train_targets = train_data.smiles(), train_data.targets()
        scaler = StandardScaler().fit(train_targets)
        scaled_targets = scaler.transform(train_targets).tolist()
        train_data.set_targets(scaled_targets)
    else:
        scaler = None

    # Get loss and metric functions
    loss_func = get_loss_func(args)
    metric_func = get_metric_func(metric=args.metric)

    # Set up test set evaluation
    test_smiles, test_targets = test_data.smiles(), test_data.targets()
    if args.dataset_type == 'multiclass':
        sum_test_preds = np.zeros((len(test_smiles), args.num_tasks, args.multiclass_num_classes))
    else:
        sum_test_preds = np.zeros((len(test_smiles), args.num_tasks))

    # Train ensemble of models
    for model_idx in range(args.ensemble_size):
        # Tensorboard writer
        save_dir = os.path.join(args.save_dir, f'model_{model_idx}')
        makedirs(save_dir)
        try:
            writer = SummaryWriter(log_dir=save_dir)
        except:
            writer = SummaryWriter(logdir=save_dir)
        # Load/build model
        if args.checkpoint_paths is not None:
            debug(f'Loading model {model_idx} from {args.checkpoint_paths[model_idx]}')
            model = load_checkpoint(args.checkpoint_paths[model_idx], current_args=args, logger=logger)
        else:
            debug(f'Building model {model_idx}')
            model = build_model(args)

        debug(model)
        debug(f'Number of parameters = {param_count(model):,}')
        if args.cuda:
            debug('Moving model to cuda')
            model = model.cuda()

        # Ensure that model is saved in correct location for evaluation if 0 epochs
        save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler, args)

        # Optimizers
        optimizer = build_optimizer(model, args)

        # Learning rate schedulers
        scheduler = build_lr_scheduler(optimizer, args)

        # Run training
        best_score = float('inf') if args.minimize_score else -float('inf')
        best_epoch, n_iter = 0, 0
        for epoch in range(args.epochs):
            debug(f'Epoch {epoch}')

            n_iter = train(
                model=model,
                data=train_data,
                loss_func=loss_func,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
                n_iter=n_iter,
                logger=logger,
                writer=writer
            )
            if isinstance(scheduler, ExponentialLR):
                scheduler.step()
            val_scores = evaluate(
                model=model,
                data=val_data,
                num_tasks=args.num_tasks,
                metric_func=metric_func,
                batch_size=args.batch_size,
                dataset_type=args.dataset_type,
                scaler=scaler,
                logger=logger
            )

            # Average validation score
            avg_val_score = np.nanmean(val_scores)
            debug(f'Validation {args.metric} = {avg_val_score:.6f}')
            writer.add_scalar(f'validation_{args.metric}', avg_val_score, n_iter)

            if args.show_individual_scores:
                # Individual validation scores
                for task_name, val_score in zip(args.task_names, val_scores):
                    debug(f'Validation {task_name} {args.metric} = {val_score:.6f}')
                    writer.add_scalar(f'validation_{task_name}_{args.metric}', val_score, n_iter)

            # Save model checkpoint if improved validation score
            if args.minimize_score and avg_val_score < best_score or \
                    not args.minimize_score and avg_val_score > best_score:
                best_score, best_epoch = avg_val_score, epoch
                save_checkpoint(os.path.join(save_dir, 'model.pt'), model, scaler, features_scaler, args)

                # Evaluate on test set using model with best validation score
        info(f'Model {model_idx} best validation {args.metric} = {best_score:.6f} on epoch {best_epoch}')
        model = load_checkpoint(os.path.join(save_dir, 'model.pt'), cuda=args.cuda, logger=logger)

        train_encoder_output, train_preds = predict(
            model=model,
            data=train_data,
            batch_size=args.batch_size,
            scaler=scaler
        )
        # torch.save(train_encoder_output, os.path.join(save_dir, 'train_encoder_output.pt'))
        train_encoder_output.to_csv(os.path.join(save_dir, 'train_encoder_output.csv'))

        val_encoder_output, validate_preds = predict(
            model=model,
            data=val_data,
            batch_size=args.batch_size,
            scaler=scaler
        )
        # torch.save(val_encoder_output, os.path.join(save_dir, 'val_encoder_output.pt'))
        val_encoder_output.to_csv(os.path.join(save_dir, 'val_encoder_output.csv'))

        test_encoder_output, test_preds = predict(
            model=model,
            data=test_data,
            batch_size=args.batch_size,
            scaler=scaler
        )
        # torch.save(test_encoder_output, os.path.join(save_dir, 'test_encoder_output.pt'))
        test_encoder_output.to_csv(os.path.join(save_dir, 'test_encoder_output.csv'))


        if len(test_preds) != 0:
            sum_test_preds += np.array(test_preds)

        mae_metric = get_metric_func(metric='mae')
        mse_metric = get_metric_func(metric='mse')
        res = []
        funcs = {'mae': mae_metric, 'mse': mse_metric, args.metric: metric_func}
        for name, func in funcs.items():
            test_scores = evaluate_predictions(
                preds=test_preds,
                targets=test_targets,
                num_tasks=args.num_tasks,
                metric_func=func,
                dataset_type=args.dataset_type,
                logger=logger
            )
            res.append(test_scores)

            # Average test score
            avg_test_score = np.nanmean(test_scores)
            info(f'Model {model_idx} test {name} = {avg_test_score:.6f}')
            writer.add_scalar(f'test_{name}', avg_test_score, 0)

            if args.show_individual_scores:
                # Individual test scores
                for task_name, test_score in zip(args.task_names, test_scores):
                    info(f'Model {model_idx} test {task_name} {name} = {test_score:.6f}')
                    writer.add_scalar(f'test_{task_name}_{name}', test_score, n_iter)

    # Evaluate ensemble on test set
    avg_test_preds = (sum_test_preds / args.ensemble_size).tolist()
    for name, func in funcs.items():
        ensemble_scores = evaluate_predictions(
            preds=avg_test_preds,
            targets=test_targets,
            num_tasks=args.num_tasks,
            metric_func=func,
            dataset_type=args.dataset_type,
            logger=logger
        )

        # Average ensemble score
        avg_ensemble_test_score = np.nanmean(ensemble_scores)
        info(f'Ensemble test {name} = {avg_ensemble_test_score:.6f}')
        writer.add_scalar(f'ensemble_test_{name}', avg_ensemble_test_score, 0)

        # Individual ensemble scores
        if args.show_individual_scores:
            for task_name, ensemble_score in zip(args.task_names, ensemble_scores):
                info(f'Ensemble test {task_name} {name} = {ensemble_score:.6f}')

    return ensemble_scores
