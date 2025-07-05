import torch.nn as nn
from torch.nn import Linear
from torch_geometric.nn import global_add_pool
from HIL import HIL
from SaTransformer import SaTransformerNetwork
import torch

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