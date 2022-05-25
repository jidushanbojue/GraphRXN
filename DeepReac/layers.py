import torch
from torch import nn
from torch.nn import functional as F
import dgl
import dgl.function as fn
from dgl.nn.pytorch import WeightAndSum
# from edgesoftmax import edge_softmax
from dgl.nn.functional import edge_softmax

def expand_as_pair(input_):
    
    if isinstance(input_, tuple):
        return input_
    else:
        return input_, input_

class Identity(nn.Module):
    
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        
        return x

def squash(s, dim=1):
    sq = torch.sum(s ** 2, dim=dim, keepdim=True)
    s_norm = torch.sqrt(sq)
    s = (sq / (1.0 + sq)) * (s / s_norm)
    return s

class WeightedSumAndMax(nn.Module):
    def __init__(self, in_feats):
        super(WeightedSumAndMax, self).__init__()
        self.in_feats = in_feats
        self.weight_and_sum = WeightAndSum(in_feats)

    def forward(self, bg, feats):

        h_g_sum = self.weight_and_sum(bg, feats)
        with bg.local_scope():
            bg.ndata['h'] = feats
            h_g_max = dgl.max_nodes(bg, 'h')
        h_g = torch.cat([h_g_sum, h_g_max], dim=1)
        return h_g

class ConditionEmbedding(nn.Module):
    
    def __init__(self, dim=16, type_num=8, pre_train=None):
        super(ConditionEmbedding, self).__init__()

        self._dim = dim
        self._type_num = type_num
        if pre_train is not None:
            self.embedding = nn.Embedding.from_pretrained(pre_train, padding_idx=0)
        else:
            self.embedding = nn.Embedding(type_num, dim, padding_idx=0)

    def forward(self, conditions):
        return self.embedding(conditions)

class GATConv(nn.Module):
    
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else: # bipartite graph neural networks
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def forward(self, graph, feat, adj, idx):
        
        graph = graph.local_var()
        if isinstance(feat, tuple):
            h_src = self.feat_drop(feat[0])
            h_dst = self.feat_drop(feat[1])
            feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
            feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
        else:
            h_src = h_dst = self.feat_drop(feat)
            feat_src = feat_dst = self.fc(h_src).view(-1, self._num_heads, self._out_feats)
        
        el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
        er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
        graph.srcdata.update({'ft': feat_src, 'el': el})
        graph.dstdata.update({'er': er})
        # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
        graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
        e = self.leaky_relu(graph.edata.pop('e'))
        # compute softmax
        if adj is not None:
            edata = self.attn_drop(edge_softmax(graph, e))
            edata_ = torch.tensor(edata)
            for i in range(len(graph.edges()[0])):
                idx1 = int(graph.edges()[0][i])
                idx2 = int(graph.edges()[1][i])
                edata_[i] = edata[i]*adj[idx1+idx,idx2+idx]
            graph.edata['a'] = edata_
        else:    
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
        # message passing
        graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                         fn.sum('m', 'ft'))
        rst = graph.dstdata['ft']
        # residual
        if self.res_fc is not None:
            resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
            rst = rst + resval
        # activation
        if self.activation:
            rst = self.activation(rst)
        return rst

class GATLayer(nn.Module):
    
    def __init__(self, in_feats, out_feats, num_heads, feat_drop, attn_drop,
                 alpha=0.2, residual=True, agg_mode='flatten', activation=None):
        super(GATLayer, self).__init__()

        self.gat_conv = GATConv(in_feats=in_feats, out_feats=out_feats, num_heads=num_heads,
                                feat_drop=feat_drop, attn_drop=attn_drop,
                                negative_slope=alpha, residual=residual)
        assert agg_mode in ['flatten', 'mean']
        self.agg_mode = agg_mode
        self.activation = activation

    def forward(self, bg, feats, adj=None, idx=None):
        
        feats = self.gat_conv(bg, feats, adj, idx)
        if self.agg_mode == 'flatten':
            feats = feats.flatten(1)
        else:
            feats = feats.mean(1)

        if self.activation is not None:
            feats = self.activation(feats)

        return feats

class GAT_adj(nn.Module):
    
    def __init__(self, in_feats, hidden_feats=None, num_heads=None, feat_drops=None,
                 attn_drops=None, alphas=None, residuals=None, agg_modes=None, activations=None):
        super(GAT_adj, self).__init__()

        if hidden_feats is None:
            hidden_feats = [32, 32]

        n_layers = len(hidden_feats)
        if num_heads is None:
            num_heads = [4 for _ in range(n_layers)]
        if feat_drops is None:
            feat_drops = [0. for _ in range(n_layers)]
        if attn_drops is None:
            attn_drops = [0. for _ in range(n_layers)]
        if alphas is None:
            alphas = [0.2 for _ in range(n_layers)]
        if residuals is None:
            residuals = [True for _ in range(n_layers)]
        if agg_modes is None:
            agg_modes = ['flatten' for _ in range(n_layers - 1)]
            agg_modes.append('mean')
        if activations is None:
            activations = [F.elu for _ in range(n_layers - 1)]
            activations.append(None)
        lengths = [len(hidden_feats), len(num_heads), len(feat_drops), len(attn_drops),
                   len(alphas), len(residuals), len(agg_modes), len(activations)]
        assert len(set(lengths)) == 1, 'Expect the lengths of hidden_feats, num_heads, ' \
                                       'feat_drops, attn_drops, alphas, residuals, ' \
                                       'agg_modes and activations to be the same, ' \
                                       'got {}'.format(lengths)
        self.hidden_feats = hidden_feats
        self.num_heads = num_heads
        self.agg_modes = agg_modes
        self.gnn_layers = nn.ModuleList()
        for i in range(n_layers):
            self.gnn_layers.append(GATLayer(in_feats, hidden_feats[i], num_heads[i],
                                            feat_drops[i], attn_drops[i], alphas[i],
                                            residuals[i], agg_modes[i], activations[i]))
            if agg_modes[i] == 'flatten':
                in_feats = hidden_feats[i] * num_heads[i]
            else:
                in_feats = hidden_feats[i]

    def forward(self, g, feats, adj=None, idx=None):
        
        for gnn in self.gnn_layers:
            feats = gnn(g, feats, adj, idx)
        return feats

class CapsuleLayer(nn.Module):
    def __init__(self, in_nodes_dim=8, in_nodes=4, out_nodes=1, out_nodes_dim=16, device='cuda'):
        super(CapsuleLayer, self).__init__()
        self.device = device
        self.in_nodes_dim, self.out_nodes_dim = in_nodes_dim, out_nodes_dim
        self.in_nodes, self.out_nodes = in_nodes, out_nodes
        self.weight = nn.Parameter(torch.randn(in_nodes, out_nodes, out_nodes_dim, in_nodes_dim))

    def forward(self, x):
        self.batch_size = x.size(0)
        u_hat = self.compute_uhat(x)
        routing = RoutingLayer(self.in_nodes, self.out_nodes, self.out_nodes_dim, batch_size=self.batch_size,
                                  device=self.device)
        routing(u_hat, routing_num=3)
        out_nodes_feature = routing.g.nodes[routing.out_indx].data['v']
        return out_nodes_feature.transpose(0, 1).unsqueeze(1).unsqueeze(4).squeeze(1)

    def compute_uhat(self, x):
        
        x = torch.stack([x] * self.out_nodes, dim=2).unsqueeze(4)
        W = self.weight.expand(self.batch_size, *self.weight.size())
        u_hat = torch.matmul(W, x).permute(1, 2, 0, 3, 4).squeeze().contiguous()
        return u_hat.view(-1, self.batch_size, self.out_nodes_dim)

class RoutingLayer(nn.Module):
    def __init__(self, in_nodes, out_nodes, f_size, batch_size=0, device='cuda'):
        super(RoutingLayer, self).__init__()
        self.batch_size = batch_size
        self.g = init_graph(in_nodes, out_nodes, f_size, device=device)
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes
        self.in_indx = list(range(in_nodes))
        self.out_indx = list(range(in_nodes, in_nodes + out_nodes))
        self.device = device

    def forward(self, u_hat, routing_num=1):
        self.g.edata['u_hat'] = u_hat
        # batch_size = self.batch_size

        # self.g.update_all(cap_message, cap_reduce)

        for r in range(routing_num):
            edges_b = self.g.edata['b'].view(self.in_nodes, self.out_nodes)
            self.g.edata['c'] = F.softmax(edges_b, dim=1).view(-1, 1)

            if self.batch_size:
                self.g.edata['c u_hat'] = self.g.edata['c'].unsqueeze(1) * self.g.edata['u_hat']
            else:
                self.g.edata['c u_hat'] = self.g.edata['c'] * self.g.edata['u_hat']

            self.g.update_all(fn.copy_e('c u_hat', 'm'), fn.sum('m', 's'))

            # self.g.update_all()

            if self.batch_size:
                self.g.nodes[self.out_indx].data['v'] = squash(self.g.nodes[self.out_indx].data['s'], dim=2)
            else:
                self.g.nodes[self.out_indx].data['v'] = squash(self.g.nodes[self.out_indx].data['s'], dim=1)

            v = torch.cat([self.g.nodes[self.out_indx].data['v']] * self.in_nodes, dim=0)
            if self.batch_size:
                self.g.edata['b'] = self.g.edata['b'] + (self.g.edata['u_hat'] * v).mean(dim=1).sum(dim=1, keepdim=True)
            else:
                self.g.edata['b'] = self.g.edata['b'] + (self.g.edata['u_hat'] * v).sum(dim=1, keepdim=True)


# class RoutingLayer(nn.Module):
#     def __init__(self, in_nodes, out_nodes, f_size, batch_size=0, device='cuda'):
#         super(RoutingLayer, self).__init__()
#         self.batch_size = batch_size
#         self.g = init_graph(in_nodes, out_nodes, f_size, device=device)
#         self.in_nodes = in_nodes
#         self.out_nodes = out_nodes
#         self.in_indx = list(range(in_nodes))
#         self.out_indx = list(range(in_nodes, in_nodes + out_nodes))
#         self.device = device
#
    # def forward(self, u_hat, routing_num=1):
    #     self.g.edata['u_hat'] = u_hat
    #     batch_size = self.batch_size
    #
    #     def cap_message(edges):
    #         if batch_size:
    #             return {'m': edges.data['c'].unsqueeze(1) * edges.data['u_hat']}
    #         else:
    #             return {'m': edges.data['c'] * edges.data['u_hat']}
    #
    #     self.g.register_message_func(cap_message)
    #
    #
    #     def cap_reduce(nodes):
    #         return {'s': torch.sum(nodes.mailbox['m'], dim=1)}
    #
    #     self.g.register_reduce_func(cap_reduce)
    #     self.g.update_all()

#         for r in range(routing_num):
#             edges_b = self.g.edata['b'].view(self.in_nodes, self.out_nodes)
#             self.g.edata['c'] = F.softmax(edges_b, dim=1).view(-1, 1)
#
#             self.g.update_all()
#
#             if self.batch_size:
#                 self.g.nodes[self.out_indx].data['v'] = squash(self.g.nodes[self.out_indx].data['s'], dim=2)
#             else:
#                 self.g.nodes[self.out_indx].data['v'] = squash(self.g.nodes[self.out_indx].data['s'], dim=1)
#
#             v = torch.cat([self.g.nodes[self.out_indx].data['v']] * self.in_nodes, dim=0)
#             if self.batch_size:
#                 self.g.edata['b'] = self.g.edata['b'] + (self.g.edata['u_hat'] * v).mean(dim=1).sum(dim=1, keepdim=True)
#             else:
#                 self.g.edata['b'] = self.g.edata['b'] + (self.g.edata['u_hat'] * v).sum(dim=1, keepdim=True)
import warnings
warnings.filterwarnings('ignore')
def init_graph(in_nodes, out_nodes, f_size, device='cuda'):
    g = dgl.DGLGraph()
    g.set_n_initializer(dgl.frame.zero_initializer)
    all_nodes = in_nodes + out_nodes
    g.add_nodes(all_nodes)
    in_indx = list(range(in_nodes))
    out_indx = list(range(in_nodes, in_nodes + out_nodes))
    # add edges use edge broadcasting
    for u in in_indx:
        g.add_edges(u, out_indx)
    g = g.to(device)
    g.edata['b'] = torch.zeros(in_nodes * out_nodes, 1).to(device)
    return g