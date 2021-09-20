import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
import dgl.nn.pytorch as dglnn
import dgl
import dgl.data
from model.rgat_layer import HeteroGraphAttentionConv
import logging
import dgl.function as fn
from dgl.nn import GATConv

logger = logging.getLogger(__name__)

class GAT(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hidden,
                 heads,
                 feat_drop=0,
                 attn_drop=0,
                 negative_slope=0.2,
                 activation=None,
                 residual=False):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, None, allow_zero_in_degree=True))

    def forward(self, g, inputs):
        h = inputs
        h = self.gat_layers[0](g, h).mean(1)
        return h



class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()

        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        h = self.conv1(graph, inputs)
        return h

class RGCN_2(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()

        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        # inputs is features of nodes
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h

class RGAT(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names, att_out_feats, relation_agg="sum"):
        super().__init__()

        logger.warning("rel_names: {}".format(rel_names))
        self.conv1 = HeteroGraphAttentionConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
                for rel in rel_names}, aggregate=relation_agg, att_in_feats=out_feats, att_out_feats=att_out_feats)
        self.conv2 = HeteroGraphAttentionConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate=relation_agg, att_in_feats=out_feats, att_out_feats=att_out_feats)

    def forward(self, graph, inputs):
        # inputs is features of nodes
        h = self.conv1(graph, inputs)
        return h

class HeteroClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, rel_names):
        super().__init__()

        self.rgcn = RGCN(in_dim, hidden_dim, hidden_dim, rel_names)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g):
        h = g.ndata['feat']
        h = self.rgcn(g, h)
        with g.local_scope():
            g.ndata['h'] = h
            # Calculate graph representation by average readout.
            hg = 0
            for ntype in g.ntypes:
                hg = hg + dgl.mean_nodes(g, 'h', ntype=ntype)
            return self.classify(hg)

def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    batched_labels = torch.tensor(labels)
    return batched_graph, batched_labels