import argparse
import numpy as np
import pandas as pd
import os
import random
import json
import copy
import torch
import torch.nn.functional as F
import dgl
from dgllife.utils import CanonicalAtomFeaturizer, mol_to_bigraph
from rdkit import Chem
from sklearn.metrics import r2_score, mean_absolute_error
import torch.optim as optim
from scipy import stats

def getarrindices(arr, indices):
    return [arr[i] for i in indices]

def get_split(length,num):
    split = []
    l = [i for i in range(length)]
    random.shuffle(l)
    for i in range(num):
        one_list = l[round(i / num * len(l)):round((i + 1) / num * len(l))]
        split.append(one_list)
    train = []
    test = []
    for j in range(num):
        test.append(split[j])
        train_ = []
        for k in range(num):
            if k!=j:
                train_ += split[k]
        train.append(train_)

    return train,test

def build_optimizer(args, params):
    filter_fn = filter(lambda p : p.requires_grad, params)
    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95, weight_decay=args.weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=args.weight_decay)

    return optimizer

def name2g(data_path, name):
    path = os.path.join(data_path,name+'.sdf')
    for mol in Chem.SDMolSupplier(path):
        g = mol_to_bigraph(mol,node_featurizer=CanonicalAtomFeaturizer())
    return g

def name2smi(data_path, name):
    path = os.path.join(data_path,name+'.sdf')
    for mol in Chem.SDMolSupplier(path):
        try:
        # g = mol_to_bigraph(mol,node_featurizer=CanonicalAtomFeaturizer())
            smi = Chem.MolToSmiles(mol, canonical=False)
        except:
            smi = ""
    return smi

def smi2graph(smi):
    if smi != "":
        mol = Chem.MolFromSmiles(smi)
        g = mol_to_bigraph(mol, node_featurizer=CanonicalAtomFeaturizer())
    else:
        g = mol_to_bigraph(None, node_featurizer=CanonicalAtomFeaturizer())
    return g

def check_smi(smi):
    try:
    # g = mol_to_bigraph(mol,node_featurizer=CanonicalAtomFeaturizer())
        smi = Chem.MolToSmiles(Chem.MolFromSmiles(smi), canonical=False)
    except:
        smi = ""
    return smi


class Meter(object):
    """Track and summarize model performance on a dataset for
    (multi-label) binary classification."""
    def __init__(self):
        self.mask = []
        self.y_pred = []
        self.y_true = []

    def update(self, y_pred, y_true, mask):
        """Update for the result of an iteration
        Parameters
        ----------
        y_pred : float32 tensor
            Predicted molecule labels with shape (B, T),
            B for batch size and T for the number of tasks
        y_true : float32 tensor
            Ground truth molecule labels with shape (B, T)
        mask : float32 tensor
            Mask for indicating the existence of ground
            truth labels with shape (B, T)
        """
        self.y_pred.append(y_pred.detach().cpu())
        self.y_true.append(y_true.detach().cpu())
        self.mask.append(mask.detach().cpu())

    def rmse(self):
        """Compute RMSE for each task.
        Returns
        -------
        list of float
            rmse for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        n_data, n_tasks = y_true.shape
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0]
            task_y_pred = y_pred[:, task][task_w != 0]
            scores.append(np.sqrt(F.mse_loss(task_y_pred, task_y_true).cpu().item()))
        return scores
    
    def r2(self):
        """Compute R2_score for each task.
        Returns
        -------
        list of float
            r2_score for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        n_data, n_tasks = y_true.shape
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            scores.append(r2_score(task_y_pred, task_y_true))
        return scores

    def mae(self):
        """Compute MAE for each task.
        Returns
        -------
        list of float
            r2_score for all tasks
        """
        mask = torch.cat(self.mask, dim=0)
        y_pred = torch.cat(self.y_pred, dim=0)
        y_true = torch.cat(self.y_true, dim=0)
        n_data, n_tasks = y_true.shape
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0].numpy()
            task_y_pred = y_pred[:, task][task_w != 0].numpy()
            scores.append(mean_absolute_error(task_y_pred, task_y_true))
        return scores

    def compute_metric(self, metric_name, reduction='mean'):
        
        if metric_name == 'rmse':
            return self.rmse()
        elif metric_name == 'r2':
            return self.r2()
        elif metric_name == 'mae':
            return self.mae()

class EarlyStopping(object):
    """Early stop performing
    Parameters
    ----------
    mode : str
        * 'higher': Higher metric suggests a better model
        * 'lower': Lower metric suggests a better model
    patience : int
        Number of epochs to wait before early stop
        if the metric stops getting improved
    filename : str or None
        Filename for storing the model checkpoint
    """
    def __init__(self, mode='higher', patience=10, filename=None):
        # if filename is None:
        #     dt = datetime.datetime.now()
        #     filename = 'early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(
        #         dt.date(), dt.hour, dt.minute, dt.second)

        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower

        self.patience = patience
        self.counter = 0
        self.filename = filename
        self.best_score = None
        self.early_stop = False

    def _check_higher(self, score, prev_best_score):
        return (score > prev_best_score)

    def _check_lower(self, score, prev_best_score):
        return (score < prev_best_score)

    def step(self, score, model, filename=None, args=None):
        if self.best_score is None:
            self.best_score = score
            if filename is not None:
                self.save_checkpoint(model, filename, args)
        elif self._check(score, self.best_score):
            self.best_score = score
            if filename is not None:
                self.save_checkpoint(model, filename, args)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, model, filename, args):
        '''Saves model when the metric on the validation set gets improved.'''
        torch.save({'model_state_dict': model.state_dict()}, os.path.join(args.ckptdir,filename+".pth"))

    # def load_checkpoint(self, model):
    #     '''Load model saved with early stopping.'''
    #     model.load_state_dict(torch.load(self.filename)['model_state_dict'])

def g2bg(graphs):
    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    return bg

def collate_molgraphs(data):
    
    if len(data[0]) == 4:
        index, graphs, names, labels = map(list, zip(*data))
        conditions = []
    elif len(data[0]) == 5:
        index, graphs, names, conditions,labels = map(list, zip(*data))
        conditions = torch.cat(conditions)
    graphs = np.array(graphs).T
    bg = [g2bg(graph) for graph in graphs]
    labels = torch.stack(labels, dim=0)
    masks = torch.ones(labels.shape)
    return index, bg, labels, masks, conditions

def collate_molgraphs_my(data):

    if len(data[0]) == 5:
        index, graphs, smiles, names, labels = map(list, zip(*data))
        conditions = []
    elif len(data[0]) == 6:
        index, graphs, smiles, names, conditions, labels = map(list, zip(*data))
        conditions = torch.cat(conditions)
    graphs = np.array(graphs).T
    bg = [g2bg(graph) for graph in graphs]
    labels = torch.stack(labels, dim=0)
    masks = torch.ones(labels.shape)

    return index, bg, labels, masks, conditions

def collate_molgraphs_new_copy(data):
    index, graphs, smiles, labels = map(list, zip(*data))
    conditions = []
    graphs = np.array(graphs).T
    bg = [g2bg(graph) for graph in graphs]
    labels = torch.stack(labels, dim=0)
    masks = torch.ones(labels.shape)
    conditions = torch.cat(conditions)

    return index, bg, labels, masks, conditions

def collate_molgraphs_new(data):
    index, graphs, smiles, labels = map(list, zip(*data))
    conditions = []
    graphs = np.array(graphs).T
    mask_fail_mol = [torch.tensor(graph.nonzero()[0],dtype=torch.long) for graph in graphs]
    graphs_new = [graph[np.where(graph != None)[0]] for graph in graphs]
    bg = [g2bg(graphs_new) for graphs_new in graphs_new]
    labels = torch.stack(labels, dim=0)
    masks = torch.ones(labels.shape)

    return index, bg, labels, masks, conditions, mask_fail_mol

def load_dataset(name):
    if name == "DatasetA":
        from Data.DatasetA import main_test
        c_num = 0
        plate1 = main_test.plate1
        plate2 = main_test.plate2
        plate3 = main_test.plate3
        unscaled = pd.read_csv('../Data/DatasetA/dataset.csv')
        raw = unscaled.values
        y = raw[:,-1]
        path = "../Data/DatasetA/sdf/"
        reactions = []
        reactions_smi_list = []
        names = []
        plates = [plate1,plate2,plate3]
        for plate in plates:
            for r in range(plate.rows):
                for c in range(plate.cols):
                    cond = plate.layout[r][c].conditions
                    g1 = name2g(path,cond['additive'])
                    g2 = name2g(path,cond['ligand'])
                    g3 = name2g(path,cond['aryl_halide'])
                    g4 = name2g(path,cond['base'])

                    g1_smi = name2smi(path, cond['additive'])
                    g2_smi = name2smi(path, cond['ligand'])
                    g3_smi = name2smi(path, cond['aryl_halide'])
                    g4_smi = name2smi(path, cond['base'])

                    reactions_smi = [g1_smi, g2_smi, g3_smi, g4_smi]
                    reactions_smi_list.append(reactions_smi)

                    name = [cond['additive'],cond['ligand'],cond['aryl_halide'],cond['base']]
                    reaction = [g1,g2,g3,g4]
                    reactions.append(reaction)
                    names.append(name)
        nan_list = [696, 741, 796, 797, 884]
        index_list = []
        for i in range(3960):
            if i not in nan_list:
                index_list.append(i)
        data = []
        for i in index_list:
            label = torch.tensor([y[i]*0.01])
            data_ = (str(i), reactions[i], reactions_smi_list[i], names[i], label)
            data.append(data_)

    elif name == "DatasetC":
        c_num = 0
        path = "../Data/DatasetC/sdf/"
        raw_data = pd.read_csv("../Data/DatasetC/dataset_D.csv")
        X = raw_data.values[:,0]
        y = (raw_data.values[:,-1])*0.01
        y = y.astype(float)
        labels = np.log((1+y)/(1-y))*0.001987*298
        data = []
        for i in range(len(y)):
            name = X[i].split('_')
            name[2] = '0'+ name[2]
            reaction = [name2g(path,i) for i in name]
            reaction_smi = [name2smi(path, i) for i in name]
            label = torch.tensor([labels[i]])
            name[0] = name[0]+name[1]
            name.pop(1)
            # data_ = (str(i),reaction,name,label)
            data_ = (str(i), reaction, reaction_smi, name, label)
            data.append(data_)

    elif name == "DatasetB":
        raw_data = pd.read_excel("../Data/DatasetB/aap9112_Data_File_S1.xlsx")
        react1 = raw_data['Reactant_1_Short_Hand'].values
        react2 = raw_data['Reactant_2_Name'].values
        ligand = raw_data['Ligand_Short_Hand'].values
        reagent = raw_data['Reagent_1_Short_Hand'].values
        solvent = raw_data['Solvent_1_Short_Hand'].values
        y = raw_data['Product_Yield_PCT_Area_UV'].values
        reagent_type = list(set(reagent))
        reagent_type.sort()
        reagent_type.reverse()
        c_num = len(reagent_type)
        data = []
        path = "../Data/DatasetB/sdf/"
        for i in range(len(y)):
            name1 = react1[i].split(',')
            name2 = react2[i].split(',')
            if name2[0] == "2d":
                continue
            ligand_ = ligand[i].replace(" ","")
            name = [name1[0],name2[0],ligand_,solvent[i]]
            reaction = [name2g(path,i) for i in name]
            reaction_smi = [name2smi(path, i) for i in name]
            name.append(reagent[i])
            condition = torch.tensor([int(reagent_type.index(reagent[i]))])
            label = torch.tensor([y[i]*0.01])
            data_ = (str(i),reaction, reaction_smi, name,condition,label)
            data.append(data_)

    return data, c_num


def load_data(src_file, y_standard=1):
    df = pd.read_csv(src_file)
    data = []
    c_num = 0
    for idx, line in df.iterrows():
        # print(idx)
        reaction = [smi2graph(smi) for smi in line['reaction'].split('*')]
        reaction_smi = [smi if reaction[i] is not None else None for i, smi in enumerate(line['reaction'].split('*'))]

        label = torch.tensor([df.loc[idx, 'output']*y_standard], dtype=torch.float32)
        data_ = (str(idx), reaction, reaction_smi, label)
        data.append(data_)

    return data, c_num


# def load_data(src_file):
#     df = pd.read_csv(src_file)[:1000]
#     data = []
#     c_num = 1
#     condition_list = [torch.tensor([0])]
#     for idx, line in df.iterrows():
#         print(idx)
#         reaction_smi = []
#         reaction = []
#         for smi in line['reaction'].split('*'):
#
#             if smi != '':
#                 reaction_smi.append(smi)
#                 reaction.append(smi2graph(smi))
#                 condition = torch.tensor([0])
#             else:
#                 condition = torch.tensor([1])
#
#             label = torch.tensor(df['output'], dtype=torch.float32)
#             data_ = (str(idx), reaction, reaction_smi, condition, label)
#             data.append(data_)
#
#
#     return data, c_num

# def load_data(src_file):
#     df = pd.read_csv(src_file)[:1000]
#     data = []
#     c_num = 0
#     for idx, line in df.iterrows():
#         print(idx)
#         try:
#             reaction_smi = [smi for smi in line['reaction'].split('*') if smi != '']
#             reaction = [smi2graph(smi) for smi in reaction_smi]
#             label = torch.tensor(df['output'], dtype=torch.float32)
#             data_ = (str(idx), reaction, reaction_smi, label)
#             data.append(data_)
#         except ValueError as e:
#             print(line['reaction'].split('*'))
#
#     return data, c_num

def arg_parse():
    parser = argparse.ArgumentParser(description="reactGAT arguments.")
    parser.add_argument("--dataset", dest="dataset", help="Input dataset.")
    parser.add_argument("--ckptdir", dest="ckptdir", help="Model checkpoint directory")
    parser.add_argument("--outdir", dest="outdir", help="result directory")
    parser.add_argument("--device", dest="device", help="cpu or cuda")
    parser.add_argument("--epochs", dest="num_epochs", type=int, help="Number of epochs to train.")
    parser.add_argument("--batch", dest="batch_size", type=int, help="batch size to train.")
    parser.add_argument("--patience", dest="patience", type=int, help="patience.")
    parser.add_argument('--lr', dest='lr', type=float, help='Learning rate.')
    parser.add_argument('--decay', dest='weight_decay', type=float, help='Learning rate decay ratio')
    # parser.add_argument("--metric", dest="metric_name", help="rmse or r2")
    parser.add_argument("--suffix", dest="name_suffix", help="suffix added to the output filename")
    parser.add_argument('--pre', dest='pre_ratio', type=float, help='Ratio of dataset for pre-training.')
    parser.add_argument("--select_mode", dest="select_mode", help="method to select data instances")
    parser.add_argument("--select_num", dest="num_selected", type=int, help="Number of data instances to select.")
    parser.add_argument("--sim_num", dest="simulation_num", type=int, help="Number of rounds for simulation.")

    parser.set_defaults(
        ckptdir="ckpt",
        outdir="results",
        dataset="DatasetA",
        device=0,
        lr=0.001,
        weight_decay=0.001,
        batch_size=64,
        num_epochs=500,
        patience=100,
        # metric_name="rmse",
        name_suffix="1",
        pre_ratio=0.1,
        select_mode = "random",
        num_selected = 10,
        simulation_num = 10,
    )
    return parser.parse_args(args=[])

def Rank(outfeats, index, predictions=None, outfeats_labeled=None, label=None, select_mode="random", num_selected=10):
    
    if select_mode == "random":
        random.shuffle(index)
        update_list = index[:num_selected]

    elif select_mode == "diversity":
        similarity_list = []
        for i in range(len(outfeats)):
            s_ = torch.cosine_similarity(outfeats[i],outfeats_labeled,dim=-1)
            s_max = torch.max(s_).item()
            similarity_list.append(s_max)
        df = pd.DataFrame(zip(index,similarity_list),columns=['index','similarity'])
        df_sorted = df.sort_values(by=['similarity'],ascending=True)
        df_index = df_sorted['index'].values
        update_list = list(df_index[:num_selected])

    elif select_mode == "adversary":
        label = label.cpu().numpy().reshape(-1)
        predictions = predictions.cpu().numpy().reshape(-1)
        label_list = []
        for i in range(len(outfeats)):
            s_ = torch.cosine_similarity(outfeats[i],outfeats_labeled,dim=-1)
            idx_max = torch.argmax(s_).item()
            label_list.append(label[idx_max])
            
        diff = abs(predictions - np.array(label_list))
        df = pd.DataFrame(zip(index,diff),columns=['index','diff'])
        df_sorted = df.sort_values(by=['diff'],ascending=False)
        df_index = df_sorted['index'].values
        update_list = list(df_index[:num_selected])

    elif select_mode == "greedy":
        pred = predictions.cpu().numpy().reshape(-1)
        df = pd.DataFrame(zip(index,pred),columns=['index','pred'])
        df_sorted = df.sort_values(by=['pred'],ascending=False)
        df_index = df_sorted['index'].values
        update_list = list(df_index[:num_selected])

    elif select_mode == "balanced":
        num_selected_ = int(num_selected/2)
        label = label.cpu().numpy().reshape(-1)
        predictions = predictions.cpu().numpy().reshape(-1)
        label_list = []
        for i in range(len(outfeats)):
            s_ = torch.cosine_similarity(outfeats[i],outfeats_labeled,dim=-1)
            idx_max = torch.argmax(s_).item()
            label_list.append(label[idx_max])
        diff = abs(predictions - np.array(label_list))

        df = pd.DataFrame(zip(index,predictions),columns=['index','pred'])
        df_sorted = df.sort_values(by=['pred'],ascending=False)
        df_index = df_sorted['index'].values
        update_list = list(df_index[:num_selected_])

        df2 = pd.DataFrame(zip(index,diff),columns=['index','diff'])
        df2_sorted = df2.sort_values(by=['diff'],ascending=False)
        df2_index = list(df2_sorted['index'].values)
        i = 0
        while len(update_list) != num_selected:
            if df2_index[i] not in update_list:
                update_list.append(df2_index[i])
            i += 1

    # elif select_mode == "balanced":
    #     num_selected_ = int(num_selected/2)
    #     label = label.cpu().numpy().reshape(-1)
    #     predictions = predictions.cpu().numpy().reshape(-1)
    #     label_list = []
    #     for i in range(len(outfeats)):
    #         s_ = torch.cosine_similarity(outfeats[i],outfeats_labeled,dim=-1)
    #         idx_max = torch.argmax(s_).item()
    #         label_list.append(label[idx_max])
            
    #     diff = abs(predictions - np.array(label_list))
    #     df = pd.DataFrame(zip(index,diff),columns=['index','diff'])
    #     df_sorted = df.sort_values(by=['diff'],ascending=False)
    #     df_index = df_sorted['index'].values
    #     update_list = list(df_index[:num_selected_])

    #     df2 = pd.DataFrame(zip(index,predictions),columns=['index','pred'])
    #     df2_sorted = df2.sort_values(by=['pred'],ascending=False)
    #     df2_index = list(df2_sorted['index'].values)
    #     i = 0
    #     while len(update_list) != num_selected:
    #         if df2_index[i] not in update_list:
    #             update_list.append(df2_index[i])
    #         i += 1

    return update_list

def run_a_train_epoch(epoch, model, data_loader, loss_criterion, optimizer, args, device):
    model.train()
    train_meter = Meter()
    index = []
    outfeat_list = []
    label_list = []
    for batch_id, batch_data in enumerate(data_loader):
        index_, bg, labels, masks, conditions = batch_data
        labels = labels.float()
        index += index_
        label_list.append(labels)
        labels, masks = labels.to(device), masks.to(device)
        
        hs = []
        bgs = []
        for bg_ in bg:
            bg_c = copy.deepcopy(bg_)
            h_ = bg_c.ndata.pop('h')
            hs.append(h_)
            bgs.append(bg_c)

        prediction, out_feats = model(bgs,hs,conditions)
        outfeat_list.append(out_feats)
        train_meter.update(prediction, labels, masks)

        if len(index_) > 19:
            loss = (loss_criterion(prediction, labels) * (masks != 0).float()).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
    # total_score = np.mean(train_meter.compute_metric(args.metric_name))
    outfeats =  torch.cat(outfeat_list)
    label_all = torch.cat(label_list)
    # print('epoch {:d}/{:d}, training {} {:.4f}'.format(
    #     epoch + 1, args.num_epochs, args.metric_name, total_score))

    return outfeats, index, label_all


def run_a_train_epoch_my(epoch, model, data_loader, loss_criterion, optimizer, args, device):
    model.train()
    train_meter = Meter()
    index = []
    outfeat_list = []
    label_list = []
    for batch_id, batch_data in enumerate(data_loader):
        index_, bg, labels, masks, conditions, mask_fail_mol = batch_data
        labels = labels.float()
        index += index_
        label_list.append(labels)
        labels, masks = labels.to(device), masks.to(device)

        hs = []
        bgs = []
        for bg_ in bg:
            bg_c = copy.deepcopy(bg_)
            h_ = bg_c.ndata.pop('h')
            hs.append(h_)
            bgs.append(bg_c)

        prediction, out_feats = model(bgs, hs, conditions, mask_fail_mol=mask_fail_mol, batch_size=len(labels))
        outfeat_list.append(out_feats)
        train_meter.update(prediction, labels, masks)

        if len(index_) > 19:
            loss = (loss_criterion(prediction, labels) * (masks != 0).float()).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # total_score = np.mean(train_meter.compute_metric(args.metric_name))
    outfeats = torch.cat(outfeat_list)
    label_all = torch.cat(label_list)
    # print('epoch {:d}/{:d}, training {} {:.4f}'.format(
    #     epoch + 1, args.num_epochs, args.metric_name, total_score))

    return outfeats, index, label_all


def run_an_eval_epoch_my(model, data_loader, args, device):
    model.eval()
    eval_meter = Meter()
    index = []
    outfeat_list = []
    label_list = []
    predict_list = []
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            index_, bg, labels, masks, conditions, mask_fail_mol = batch_data
            index += index_
            label_list.append(labels)
            labels, masks = labels.to(device), masks.to(device)

            hs = []
            bgs = []
            for bg_ in bg:
                bg_c = copy.deepcopy(bg_)
                h_ = bg_c.ndata.pop('h')
                hs.append(h_)
                bgs.append(bg_c)

            prediction, out_feats = model(bgs, hs, conditions, mask_fail_mol=mask_fail_mol, batch_size=len(labels))
            outfeat_list.append(out_feats)
            predict_list.append(prediction)
            eval_meter.update(prediction, labels, masks)
        total_score = []
        for metric in ["rmse", "mae", "r2"]:
            score = np.mean(eval_meter.compute_metric(metric))
            total_score.append(float(score))
        outfeats = torch.cat(outfeat_list)
        predictions = torch.cat(predict_list)
        label_lists = torch.cat(label_list)
    return total_score, outfeats, index, label_lists, predictions


def run_an_eval_epoch(model, data_loader, args, device):
    model.eval()
    eval_meter = Meter()
    index = []
    outfeat_list = []
    label_list = []
    predict_list = []
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            index_, bg, labels, masks, conditions = batch_data
            index += index_
            label_list.append(labels)
            labels, masks = labels.to(device), masks.to(device)

            hs = []
            bgs = []
            for bg_ in bg:
                bg_c = copy.deepcopy(bg_)
                h_ = bg_c.ndata.pop('h')
                hs.append(h_)
                bgs.append(bg_c)

            prediction, out_feats = model(bgs, hs, conditions)
            outfeat_list.append(out_feats)
            predict_list.append(prediction)
            eval_meter.update(prediction, labels, masks)
        total_score = []
        for metric in ["rmse", "mae", "r2"]:
            score = np.mean(eval_meter.compute_metric(metric))
            total_score.append(float(score))
        outfeats = torch.cat(outfeat_list)
        predictions = torch.cat(predict_list)
        label_lists = torch.cat(label_list)
    return total_score, outfeats, index, label_lists, predictions


