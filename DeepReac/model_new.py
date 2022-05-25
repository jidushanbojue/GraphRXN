import torch
from torch import nn
from torch.nn import functional as F
import dgl
from layers import GATLayer, CapsuleLayer, squash, GAT_adj, WeightedSumAndMax, ConditionEmbedding

class DeepReac(nn.Module):
    
    def __init__(self,
                 in_feats_0,
                 g_num,
                 c_num=0,
                 hidden_feats_0=[32, 32],
                 num_heads_0=[4, 4],
                 dropout_0=None,
                 alphas_0=None,
                 residuals_0=None,
                 agg_modes_0=None,
                 activations_0=None,
                 hidden_feats_1 = [32,32],
                 num_heads_1=[4,4],
                 dropout_1=0,
                 dropout_2=0,
                 out_dim=32,
                 device="cuda"):
        
        super(DeepReac, self).__init__()
        
        self.g_num = g_num
        self.c_num = c_num
        self.out_dim = out_dim
        self.device = device
        
        self.gnn = GAT_adj(in_feats=in_feats_0,
                       hidden_feats=hidden_feats_0,
                       num_heads=num_heads_0,
                       feat_drops=dropout_0,
                       attn_drops=dropout_0,
                       alphas=alphas_0,
                       residuals=residuals_0,
                       agg_modes=agg_modes_0,
                       activations=activations_0)

        if self.gnn.agg_modes[-1] == 'flatten':
            gnn_out_feats = self.gnn.hidden_feats[-1] * self.gnn.num_heads[-1]
        else:
            gnn_out_feats = self.gnn.hidden_feats[-1]
        self.readout = WeightedSumAndMax(gnn_out_feats)
        
        num_layers = len(num_heads_1)
        self.gnn_layers = nn.ModuleList()
        in_feats = 2*gnn_out_feats
        
        if c_num > 0:
            self.embedding_layer = ConditionEmbedding(dim=in_feats, type_num=c_num)
            self.num = self.g_num+1
        else:
            self.num = self.g_num
        for l in range(num_layers):
            if l > 0:
                in_feats = hidden_feats_1[l - 1] * num_heads_1[l - 1]

            if l == num_layers - 1:
                agg_mode = 'mean'
                agg_act = None
            else:
                agg_mode = 'flatten'
                agg_act = F.elu

            self.gnn_layers.append(GATLayer(in_feats, hidden_feats_1[l], num_heads_1[l],
                                            feat_drop=dropout_1, attn_drop=dropout_1,
                                            agg_mode=agg_mode, activation=agg_act))

        self.in_dim = hidden_feats_1[-1]
        self.digits = CapsuleLayer(in_nodes_dim=self.in_dim, in_nodes=self.num, out_nodes=2, out_nodes_dim=self.out_dim, device=self.device)
        
        self.predict = nn.Sequential(
            nn.Dropout(dropout_2),
            nn.Linear(self.out_dim*2, 1)
        )
    
    def get_bg(self, batch_size, num):
        
        graphs = []
        node_s = []
        node_d = []
        for i in range(num):
            node_s += (num-1)*[i]
            for j in range(num):
                if j != i :
                    node_d.append(j)
        # for i in range(batch_size):
        #     G = dgl.DGLGraph()
        #     G.add_nodes(num)
        #     src = torch.tensor(node_s)
        #     dst = torch.tensor(node_d)
        #     G.add_edges(src, dst)
        #     graphs.append(G)
        for i in range(batch_size):
            src = torch.tensor(node_s)
            dst = torch.tensor(node_d)
            G = dgl.graph((src, dst))
            graphs.append(G)
        
        bg = dgl.batch(graphs)
        bg.set_n_initializer(dgl.init.zero_initializer)
        bg.set_e_initializer(dgl.init.zero_initializer)
        bg.to(self.device)
        return bg
    
    def forward(self, bgs, hs, conditions=[], mask_fail_mol=None, batch_size=64, adjs=None, wanted=None):
        
        node_feats = []
        i_ = 0
        for i in range(self.g_num):
            feats = hs[i].to(self.device)
            bg = bgs[i].to(self.device)
            mask_fail_mol_i = mask_fail_mol[i].to(self.device)
            if adjs is not None:
                if wanted[i] == 1:
                    idx = int(adjs.shape[0]/sum(wanted))*i_
                    node_feat = self.gnn(bg, feats, adjs, idx)
                    i_+=1
                else:
                    node_feat = self.gnn(bg, feats)
            else:
                node_feat = self.gnn(bg, feats)
            graph_feat = self.readout(bg, node_feat)
            graph_feat_temp = torch.zeros(batch_size, graph_feat.shape[1],device=self.device)
            # print('graph_feat old shape:', graph_feat.shape)
            graph_feat_temp = graph_feat_temp.scatter(dim=0, index=mask_fail_mol_i.unsqueeze(1).repeat(1, graph_feat.shape[1]), src=graph_feat)
            # print('graph_feat new shape:', graph_feat_temp.shape)
            node_feats.append(graph_feat_temp)

        if len(conditions) == 0:
            feats = torch.cat((node_feats),dim=1).reshape(-1,node_feats[0].shape[-1])
        else: 
            conditions = conditions.to(self.device)
            c_feats = self.embedding_layer(conditions)
            node_feats.append(c_feats)
            feats = torch.cat((node_feats),dim=1).reshape(-1,node_feats[0].shape[-1])
            
        bg = self.get_bg(bgs[0].batch_size, self.num)
        bg = bg.to(self.device)
        for gnn in self.gnn_layers:
            feats = gnn(bg, feats)

        feats = feats.reshape(-1,self.num,self.in_dim)
        feats = squash(feats, dim=2)
        g_feats = self.digits(feats).reshape(-1,self.out_dim*2)
        
        return self.predict(g_feats), g_feats

class DeepReac_noG(nn.Module):
    
    def __init__(self,
                 in_feats_0,
                 g_num,
                 c_num=0,
                 hidden_feats_0=[32, 32],
                 num_heads_0=[4, 4],
                 dropout_0=None,
                 alphas_0=None,
                 residuals_0=None,
                 agg_modes_0=None,
                 activations_0=None,
                 dropout_2=0,
                 out_dim=32,
                 device="cuda"):
        
        super(DeepReac_noG, self).__init__()
        
        self.g_num = g_num
        self.c_num = c_num
        self.out_dim = out_dim
        self.device = device
        
        self.gnn = GAT_adj(in_feats=in_feats_0,
                       hidden_feats=hidden_feats_0,
                       num_heads=num_heads_0,
                       feat_drops=dropout_0,
                       attn_drops=dropout_0,
                       alphas=alphas_0,
                       residuals=residuals_0,
                       agg_modes=agg_modes_0,
                       activations=activations_0)

        if self.gnn.agg_modes[-1] == 'flatten':
            gnn_out_feats = self.gnn.hidden_feats[-1] * self.gnn.num_heads[-1]
        else:
            gnn_out_feats = self.gnn.hidden_feats[-1]
        self.readout = WeightedSumAndMax(gnn_out_feats)
        
        self.in_dim = 2*gnn_out_feats
        
        if c_num > 0:
            self.embedding_layer = ConditionEmbedding(dim=self.in_dim, type_num=c_num)
            self.num = self.g_num+1
        else:
            self.num = self.g_num

        self.digits = CapsuleLayer(in_nodes_dim=self.in_dim, in_nodes=self.num, out_nodes=2, out_nodes_dim=self.out_dim, device=self.device)
        
        self.predict = nn.Sequential(
            nn.Dropout(dropout_2),
            nn.Linear(self.out_dim*2, 1)
        )

    def forward(self, bgs, hs, conditions=[], adjs=None, wanted=None):
        
        node_feats = []
        i_ = 0
        for i in range(self.g_num):
            feats = hs[i].to(self.device)
            bg = bgs[i].to(self.device)
            if adjs is not None:
                if wanted[i] == 1:
                    idx = int(adjs.shape[0]/sum(wanted))*i_
                    node_feat = self.gnn(bg, feats, adjs, idx)
                    i_+=1
                else:
                    node_feat = self.gnn(bg, feats)
            else:
                node_feat = self.gnn(bg, feats)
            graph_feat = self.readout(bg, node_feat)
            node_feats.append(graph_feat)

        if len(conditions) == 0:
            feats = torch.cat((node_feats),dim=1).reshape(-1,self.num,self.in_dim)
        else: 
            conditions = conditions.to(self.device)
            c_feats = self.embedding_layer(conditions)
            node_feats.append(c_feats)
            feats = torch.cat((node_feats),dim=1).reshape(-1,self.num,self.in_dim)

       # feats = squash(feats, dim=2)
        g_feats = self.digits(feats).reshape(-1,self.out_dim*2)
        
        return self.predict(g_feats), g_feats

class DeepReac_noC(nn.Module):
    
    def __init__(self,
                 in_feats_0,
                 g_num,
                 c_num=0,
                 hidden_feats_0=[32, 32],
                 num_heads_0=[4, 4],
                 dropout_0=None,
                 alphas_0=None,
                 residuals_0=None,
                 agg_modes_0=None,
                 activations_0=None,
                 hidden_feats_1 = [32,32],
                 num_heads_1=[4,4],
                 dropout_1=0,
                 dropout_2=0,
                 device="cuda"):
        
        super(DeepReac_noC, self).__init__()
        
        self.g_num = g_num
        self.c_num = c_num
        self.device = device
        
        self.gnn = GAT_adj(in_feats=in_feats_0,
                       hidden_feats=hidden_feats_0,
                       num_heads=num_heads_0,
                       feat_drops=dropout_0,
                       attn_drops=dropout_0,
                       alphas=alphas_0,
                       residuals=residuals_0,
                       agg_modes=agg_modes_0,
                       activations=activations_0)

        if self.gnn.agg_modes[-1] == 'flatten':
            gnn_out_feats = self.gnn.hidden_feats[-1] * self.gnn.num_heads[-1]
        else:
            gnn_out_feats = self.gnn.hidden_feats[-1]
        self.readout = WeightedSumAndMax(gnn_out_feats)
        
        num_layers = len(num_heads_1)
        self.gnn_layers = nn.ModuleList()
        in_feats = 2*gnn_out_feats
        
        if c_num > 0:
            self.embedding_layer = ConditionEmbedding(dim=in_feats, type_num=c_num)
            self.num = self.g_num+1
        else:
            self.num = self.g_num
        for l in range(num_layers):
            if l > 0:
                in_feats = hidden_feats_1[l - 1] * num_heads_1[l - 1]

            if l == num_layers - 1:
                agg_mode = 'mean'
                agg_act = None
            else:
                agg_mode = 'flatten'
                agg_act = F.elu

            self.gnn_layers.append(GATLayer(in_feats, hidden_feats_1[l], num_heads_1[l],
                                            feat_drop=dropout_1, attn_drop=dropout_1,
                                            agg_mode=agg_mode, activation=agg_act))
        self.in_dim = hidden_feats_1[-1]*self.num
        
        self.predict = nn.Sequential(
            nn.Dropout(dropout_2),
            nn.Linear(self.in_dim, 1)
        )
    
    def get_bg(self, batch_size, num):
        
        graphs = []
        node_s = []
        node_d = []
        for i in range(num):
            node_s += (num-1)*[i]
            for j in range(num):
                if j != i :
                    node_d.append(j)
        # for i in range(batch_size):
        #     G = dgl.DGLGraph()
        #     G.add_nodes(num)
        #     src = torch.tensor(node_s)
        #     dst = torch.tensor(node_d)
        #     G.add_edges(src, dst)
        #     graphs.append(G)
        for i in range(batch_size):
            src = torch.tensor(node_s)
            dst = torch.tensor(node_d)
            G = dgl.graph((src, dst))
            graphs.append(G)
        
        bg = dgl.batch(graphs)
        bg.set_n_initializer(dgl.init.zero_initializer)
        bg.set_e_initializer(dgl.init.zero_initializer)
        bg.to(self.device)
        return bg
    
    def forward(self, bgs, hs, conditions=[], adjs=None, wanted=None):
        
        node_feats = []
        i_ = 0
        for i in range(self.g_num):
            feats = hs[i].to(self.device)
            bg = bgs[i].to(self.device)
            if adjs is not None:
                if wanted[i] == 1:
                    idx = int(adjs.shape[0]/sum(wanted))*i_
                    node_feat = self.gnn(bg, feats, adjs, idx)
                    i_+=1
                else:
                    node_feat = self.gnn(bg, feats)
            else:
                node_feat = self.gnn(bg, feats)
            graph_feat = self.readout(bg, node_feat)
            node_feats.append(graph_feat)

        if len(conditions) == 0:
            feats = torch.cat((node_feats),dim=1).reshape(-1,node_feats[0].shape[-1])
        else: 
            conditions = conditions.to(self.device)
            c_feats = self.embedding_layer(conditions)
            node_feats.append(c_feats)
            feats = torch.cat((node_feats),dim=1).reshape(-1,node_feats[0].shape[-1])
            
        bg = self.get_bg(bgs[0].batch_size, self.num)
        for gnn in self.gnn_layers:
            feats = gnn(bg, feats)

        feats = feats.reshape(-1,self.in_dim)
        
        return self.predict(feats), feats

class DeepReac_noboth(nn.Module):
    
    def __init__(self,
                 in_feats_0,
                 g_num,
                 c_num=0,
                 hidden_feats_0=[32, 32],
                 num_heads_0=[4, 4],
                 dropout_0=None,
                 alphas_0=None,
                 residuals_0=None,
                 agg_modes_0=None,
                 activations_0=None,
                 dropout_2=0,
                 device="cuda"):
        
        super(DeepReac_noboth, self).__init__()
        
        self.g_num = g_num
        self.c_num = c_num
        self.device = device
        
        self.gnn = GAT_adj(in_feats=in_feats_0,
                       hidden_feats=hidden_feats_0,
                       num_heads=num_heads_0,
                       feat_drops=dropout_0,
                       attn_drops=dropout_0,
                       alphas=alphas_0,
                       residuals=residuals_0,
                       agg_modes=agg_modes_0,
                       activations=activations_0)

        if self.gnn.agg_modes[-1] == 'flatten':
            gnn_out_feats = self.gnn.hidden_feats[-1] * self.gnn.num_heads[-1]
        else:
            gnn_out_feats = self.gnn.hidden_feats[-1]
        self.readout = WeightedSumAndMax(gnn_out_feats)
        
        if c_num > 0:
            self.embedding_layer = ConditionEmbedding(dim=2*gnn_out_feats, type_num=c_num)
            nodes = self.g_num+1
        else:
            nodes = self.g_num
        
        feats_dim = 2*gnn_out_feats*nodes
        self.predict = nn.Sequential(
            nn.Dropout(dropout_2),
            nn.Linear(feats_dim, 1)
        )

    def forward(self, bgs, hs, conditions=[], adjs=None, wanted=None):
        
        node_feats = []
        i_ = 0
        for i in range(self.g_num):
            feats = hs[i].to(self.device)
            bg = bgs[i].to(self.device)
            if adjs is not None:
                if wanted[i] == 1:
                    idx = int(adjs.shape[0]/sum(wanted))*i_
                    node_feat = self.gnn(bg, feats, adjs, idx)
                    i_+=1
                else:
                    node_feat = self.gnn(bg, feats)
            else:
                node_feat = self.gnn(bg, feats)
            graph_feat = self.readout(bg, node_feat)
            node_feats.append(graph_feat)

        if len(conditions) == 0:
            feats = torch.cat((node_feats),dim=1)
        else: 
            conditions = conditions.to(self.device)
            c_feats = self.embedding_layer(conditions)
            node_feats.append(c_feats)
            feats = torch.cat((node_feats),dim=1)

        return self.predict(feats), feats

class GATadj_capsule_old(nn.Module):
    
    def __init__(self,
                 in_feats_0,
                 g_num,
                 c_num=0,
                 hidden_feats_0=[16, 8],
                 num_heads_0=[4, 4],
                 dropout_0=None,
                 alphas_0=None,
                 residuals_0=None,
                 agg_modes_0=None,
                 activations_0=None,
                 hidden_feats_1 = [16,8],
                 num_heads_1=[2,1],
                 dropout_1=0,
                 dropout_2=0,
                 out_dim=16,
                 device="cuda"):
        
        super(GATadj_capsule, self).__init__()
        
        self.g_num = g_num
        self.c_num = c_num
        self.out_dim = out_dim
        self.device = device
        
        self.gnn = GAT_adj(in_feats=in_feats_0,
                       hidden_feats=hidden_feats_0,
                       num_heads=num_heads_0,
                       feat_drops=dropout_0,
                       attn_drops=dropout_0,
                       alphas=alphas_0,
                       residuals=residuals_0,
                       agg_modes=agg_modes_0,
                       activations=activations_0)

        if self.gnn.agg_modes[-1] == 'flatten':
            gnn_out_feats = self.gnn.hidden_feats[-1] * self.gnn.num_heads[-1]
        else:
            gnn_out_feats = self.gnn.hidden_feats[-1]
        self.readout = WeightedSumAndMax(gnn_out_feats)
        
        num_layers = len(num_heads_1)
        self.gnn_layers = nn.ModuleList()
        in_feats = 2*gnn_out_feats
        
        if c_num > 0:
            self.embedding_layer = ConditionEmbedding(dim=in_feats, type_num=c_num)
            in_nodes = self.g_num+1
        else:
            in_nodes = self.g_num
        for l in range(num_layers):
            if l > 0:
                in_feats = hidden_feats_1[l - 1] * num_heads_1[l - 1]

            if l == num_layers - 1:
                agg_mode = 'mean'
                agg_act = None
            else:
                agg_mode = 'flatten'
                agg_act = F.elu

            self.gnn_layers.append(GATLayer(in_feats, hidden_feats_1[l], num_heads_1[l],
                                            feat_drop=dropout_1, attn_drop=dropout_1,
                                            agg_mode=agg_mode, activation=agg_act))
        self.in_dim = hidden_feats_1[-1]
        self.digits = CapsuleLayer(in_nodes_dim=self.in_dim, in_nodes=in_nodes, out_nodes_dim=self.out_dim, device=self.device)
        
        self.predict = nn.Sequential(
            nn.Dropout(dropout_2),
            nn.Linear(self.out_dim, 1)
        )
    
    def get_bg(self, batch_size, num):
        
        graphs = []
        node_s = []
        node_d = []
        for i in range(num):
            node_s += (num-1)*[i]
            for j in range(num):
                if j != i :
                    node_d.append(j)
        # for i in range(batch_size):
        #     G = dgl.DGLGraph()
        #     G.add_nodes(num)
        #     src = torch.tensor(node_s)
        #     dst = torch.tensor(node_d)
        #     G.add_edges(src, dst)
        #     graphs.append(G)
        for i in range(batch_size):
            src = torch.tensor(node_s)
            dst = torch.tensor(node_d)
            G = dgl.graph((src, dst))
            graphs.append(G)
        
        bg = dgl.batch(graphs)
        bg.set_n_initializer(dgl.init.zero_initializer)
        bg.set_e_initializer(dgl.init.zero_initializer)
        bg.to(self.device)
        return bg
    
    def forward(self, bgs, hs, conditions=[], adjs=None, wanted=None):
        
        node_feats = []
        i_ = 0
        for i in range(self.g_num):
            feats = hs[i].to(self.device)
            bg = bgs[i].to(self.device)
            if adjs is not None:
                if wanted[i] == 1:
                    idx = int(adjs.shape[0]/sum(wanted))*i_
                    node_feat = self.gnn(bg, feats, adjs, idx)
                    i_+=1
                else:
                    node_feat = self.gnn(bg, feats)
            else:
                node_feat = self.gnn(bg, feats)
            graph_feat = self.readout(bg, node_feat)
            node_feats.append(graph_feat)

        if len(conditions) == 0:
            feats = torch.cat((node_feats),dim=1).reshape(-1,node_feats[0].shape[-1])
            num = self.g_num            
        else: 
            conditions = conditions.to(self.device)
            c_feats = self.embedding_layer(conditions)
            node_feats.append(c_feats)
            feats = torch.cat((node_feats),dim=1).reshape(-1,node_feats[0].shape[-1])
            num = self.g_num + 1
            
        bg = self.get_bg(bgs[0].batch_size, num)
        for gnn in self.gnn_layers:
            feats = gnn(bg, feats)

        feats = feats.reshape(-1,num,self.in_dim)
        #feats = squash(feats, dim=2)
        g_feats = self.digits(feats).reshape(-1,self.out_dim)
        
        return self.predict(g_feats), g_feats


class DeepReac_old(nn.Module):
    
    def __init__(self,
                 in_feats_0,
                 g_num,
                 c_num=0,
                 hidden_feats_0=[32, 32],
                 num_heads_0=[4, 4],
                 dropout_0=None,
                 alphas_0=None,
                 residuals_0=None,
                 agg_modes_0=None,
                 activations_0=None,
                 hidden_feats_1 = [32,32],
                 num_heads_1=[4,4],
                 dropout_1=0,
                 dropout_2=0,
                 out_dim=64,
                 device="cuda"):
        
        super(DeepReac_old, self).__init__()
        
        self.g_num = g_num
        self.c_num = c_num
        self.out_dim = out_dim
        self.device = device
        
        self.gnn = GAT_adj(in_feats=in_feats_0,
                       hidden_feats=hidden_feats_0,
                       num_heads=num_heads_0,
                       feat_drops=dropout_0,
                       attn_drops=dropout_0,
                       alphas=alphas_0,
                       residuals=residuals_0,
                       agg_modes=agg_modes_0,
                       activations=activations_0)

        if self.gnn.agg_modes[-1] == 'flatten':
            gnn_out_feats = self.gnn.hidden_feats[-1] * self.gnn.num_heads[-1]
        else:
            gnn_out_feats = self.gnn.hidden_feats[-1]
        self.readout = WeightedSumAndMax(gnn_out_feats)
        
        num_layers = len(num_heads_1)
        self.gnn_layers = nn.ModuleList()
        in_feats = 2*gnn_out_feats
        
        if c_num > 0:
            self.embedding_layer = ConditionEmbedding(dim=in_feats, type_num=c_num)
            self.num = self.g_num+1
        else:
            self.num = self.g_num
        for l in range(num_layers):
            if l > 0:
                in_feats = hidden_feats_1[l - 1] * num_heads_1[l - 1]

            if l == num_layers - 1:
                agg_mode = 'mean'
                agg_act = None
            else:
                agg_mode = 'flatten'
                agg_act = F.elu

            self.gnn_layers.append(GATLayer(in_feats, hidden_feats_1[l], num_heads_1[l],
                                            feat_drop=dropout_1, attn_drop=dropout_1,
                                            agg_mode=agg_mode, activation=agg_act))

        self.in_dim = hidden_feats_1[-1]
        self.digits = CapsuleLayer(in_nodes_dim=self.in_dim, in_nodes=self.num, out_nodes_dim=self.out_dim, device=self.device)
        
        self.predict = nn.Sequential(
            nn.Dropout(dropout_2),
            nn.Linear(self.out_dim, 1)
        )
    
    def get_bg(self, batch_size, num):
        
        graphs = []
        node_s = []
        node_d = []
        for i in range(num):
            node_s += (num-1)*[i]
            for j in range(num):
                if j != i :
                    node_d.append(j)
        # for i in range(batch_size):
        #     G = dgl.DGLGraph()
        #     G.add_nodes(num)
        #     src = torch.tensor(node_s)
        #     dst = torch.tensor(node_d)
        #     G.add_edges(src, dst)
        #     graphs.append(G)
        for i in range(batch_size):
            src = torch.tensor(node_s)
            dst = torch.tensor(node_d)
            G = dgl.graph((src, dst))
            graphs.append(G)
        
        bg = dgl.batch(graphs)
        bg.set_n_initializer(dgl.init.zero_initializer)
        bg.set_e_initializer(dgl.init.zero_initializer)
        bg.to(self.device)
        return bg
    
    def forward(self, bgs, hs, conditions=[], adjs=None, wanted=None):
        
        node_feats = []
        i_ = 0
        for i in range(self.g_num):
            feats = hs[i].to(self.device)
            bg = bgs[i].to(self.device)
            if adjs is not None:
                if wanted[i] == 1:
                    idx = int(adjs.shape[0]/sum(wanted))*i_
                    node_feat = self.gnn(bg, feats, adjs, idx)
                    i_+=1
                else:
                    node_feat = self.gnn(bg, feats)
            else:
                node_feat = self.gnn(bg, feats)
            graph_feat = self.readout(bg, node_feat)
            node_feats.append(graph_feat)

        if len(conditions) == 0:
            feats = torch.cat((node_feats),dim=1).reshape(-1,node_feats[0].shape[-1])
        else: 
            conditions = conditions.to(self.device)
            c_feats = self.embedding_layer(conditions)
            node_feats.append(c_feats)
            feats = torch.cat((node_feats),dim=1).reshape(-1,node_feats[0].shape[-1])
            
        bg = self.get_bg(bgs[0].batch_size, self.num)
        for gnn in self.gnn_layers:
            feats = gnn(bg, feats)

        feats = feats.reshape(-1,self.num,self.in_dim)
        feats = squash(feats, dim=2)
        g_feats = self.digits(feats).reshape(-1,self.out_dim)
        
        return self.predict(g_feats), g_feats