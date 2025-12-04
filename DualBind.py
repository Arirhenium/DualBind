import torch.nn as nn
from torch.nn import Linear
from torch_geometric.nn import global_add_pool
from SaTransformer import SaTransformerNetwork
import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing

class HIL(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(HIL, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.mlp_node_cov = nn.Sequential(
            nn.Linear(self.in_channels, self.out_channels),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.out_channels))
        self.mlp_node_ncov = nn.Sequential(
            nn.Linear(self.in_channels, self.out_channels),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(self.out_channels))

        self.mlp_coord_cov = nn.Sequential(nn.Linear(9, self.in_channels), nn.SiLU())
        self.mlp_coord_ncov = nn.Sequential(nn.Linear(9, self.in_channels), nn.SiLU())

    def forward(self, x, edge_index_intra, edge_index_inter, pos=None, size=None):
        row_cov, col_cov = edge_index_intra
        coord_diff_cov = pos[row_cov] - pos[col_cov]
        radial_cov = self.mlp_coord_cov(_rbf(torch.norm(coord_diff_cov, dim=-1), D_min=0., D_max=6., D_count=9, device=x.device))
        out_node_intra = self.propagate(edge_index=edge_index_intra, x=x, radial=radial_cov, size=size)

        row_ncov, col_ncov = edge_index_inter
        coord_diff_ncov = pos[row_ncov] - pos[col_ncov]
        radial_ncov = self.mlp_coord_ncov(_rbf(torch.norm(coord_diff_ncov, dim=-1), D_min=0., D_max=6., D_count=9, device=x.device))
        out_node_inter = self.propagate(edge_index=edge_index_inter, x=x, radial=radial_ncov, size=size)

        out_node = self.mlp_node_cov(x + out_node_intra) + self.mlp_node_ncov(x + out_node_inter)
        return out_node

    def message(self, x_j: Tensor, x_i: Tensor, radial, index: Tensor):
        x = x_j * radial
        return x

def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    D_mu = torch.linspace(D_min, D_max, D_count).to(device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF

class DualBind(nn.Module):
    def __init__(self, node_dim, hidden_dim):#35，256
        super().__init__()
        self.lin_node_c = nn.Sequential(Linear(node_dim, hidden_dim), nn.SiLU())
        self.gconv1 = HIL(hidden_dim, hidden_dim)
        self.gconv2 = HIL(hidden_dim, hidden_dim)
        self.gconv3 = HIL(hidden_dim, hidden_dim)

        self.sat = SaTransformerNetwork(hidden_dim)
        self.cat_dropout = nn.Dropout(0.2)

        self.fc1 = FC(hidden_dim, hidden_dim, 3, 0.1, 1)
        self.fc2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.Dropout(0.5),
            nn.PReLU(),
            nn.Linear(128, 64),
            nn.Dropout(0.5),
            nn.PReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, data):
        data_c = data['complex']
        prot_trans,lig_trans,pkt_trans,prot_mask,smi_mask = data['prot_trans'], data['smi_trans'], data ['pkt_tensor'], data['prot_msk'], data['smi_msk']
        x_c, edge_c_intra, edge_c_inter, pos = data_c.x, data_c.edge_index_intra, data_c.edge_index_inter, data_c.pos

        x_c = self.lin_node_c(x_c)
        x_c = self.gconv1(x_c, edge_c_intra, edge_c_inter, pos)
        x_c = self.gconv2(x_c, edge_c_intra, edge_c_inter, pos)
        x_c = self.gconv3(x_c, edge_c_intra, edge_c_inter, pos)

        lig, pkt = self.sat(prot_trans, lig_trans, pkt_trans, prot_mask, smi_mask)

        pre_sat = torch.cat([lig, pkt],dim=1)
        pre_sat = self.cat_dropout(pre_sat)
        pre_sat = self.fc2(pre_sat)

        pre_com = global_add_pool(x_c, data_c.batch)
        pre_com = self.fc1(pre_com)

        return (pre_com + pre_sat).view(-1)

class FC(nn.Module):
    def __init__(self, d_graph_layer, d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(FC, self).__init__()
        self.d_graph_layer = d_graph_layer
        self.d_FC_layer = d_FC_layer
        self.n_FC_layer = n_FC_layer
        self.dropout = dropout
        self.predict = nn.ModuleList()
        for j in range(self.n_FC_layer):
            if j == 0:
                self.predict.append(nn.Linear(self.d_graph_layer, self.d_FC_layer))
                self.predict.append(nn.Dropout(self.dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(d_FC_layer))
            if j == self.n_FC_layer - 1:# last layer
                self.predict.append(nn.Linear(self.d_FC_layer, n_tasks))
            else:
                self.predict.append(nn.Linear(self.d_FC_layer, self.d_FC_layer))
                self.predict.append(nn.Dropout(self.dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(d_FC_layer))

    def forward(self, h):
        for layer in self.predict:
            h = layer(h)

        return h

# %%