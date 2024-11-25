import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
import torch.nn as nn
import torch.nn.functional as F
from egnn_clean import EGNN

# heterogeneous interaction layer
class Squeeze(nn.Module):   #Dimention Module
    def forward(self, input: torch.Tensor):
        return input.squeeze()

class DilatedConv(nn.Module):     # Dilated Convolution
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False, dilation=d)

    def forward(self, input):
        output = self.conv(input)
        return output

class DilatedConvBlockA(nn.Module):
    def __init__(self, nIn, nOut, add=True):
        super().__init__()
        n = int(nOut / 5)
        n1 = nOut - 4 * n
        self.c1 = nn.Conv1d(nIn, n, 1, padding=0)  # Down Dimention
        self.br1 = nn.Sequential(nn.BatchNorm1d(n), nn.PReLU())
        self.d1 = DilatedConv(n, n1, 3, 1, 1)    # Dilated scale:1(2^0)
        self.d2 = DilatedConv(n, n, 3, 1, 2)     # Dilated scale:2(2^1)
        self.d4 = DilatedConv(n, n, 3, 1, 4)     # Dilated scale:4(2^2)
        self.d8 = DilatedConv(n, n, 3, 1, 8)     # Dilated scale:8(2^3)
        self.d16 = DilatedConv(n, n, 3, 1, 16)   # Dilated scale:16(2^4)
        self.br2 = nn.Sequential(nn.BatchNorm1d(nOut), nn.PReLU())

        if nIn != nOut:
            add = False
        self.add = add

    def forward(self, input):
        output1 = self.c1(input)
        output1 = self.br1(output1)

        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)

        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16

        combine = torch.cat([d1, add1, add2, add3, add4], 1)

        if self.add:
            combine = input + combine
        output = self.br2(combine)
        return output


class DilatedConvBlockB(nn.Module):
    def __init__(self, nIn, nOut, add=True):
        super().__init__()
        n = int(nOut / 4)
        n1 = nOut - 3 * n
        self.c1 = nn.Conv1d(nIn, n, 1, padding=0)
        self.br1 = nn.Sequential(nn.BatchNorm1d(n), nn.PReLU())
        self.d1 = DilatedConv(n, n1, 3, 1, 1)  # Dilated scale:1(2^0)
        self.d2 = DilatedConv(n, n, 3, 1, 2)   # Dilated scale:2(2^1)
        self.d4 = DilatedConv(n, n, 3, 1, 4)   # Dilated scale:4(2^2)
        self.d8 = DilatedConv(n, n, 3, 1, 8)   # Dilated scale:8(2^3)
        self.br2 = nn.Sequential(nn.BatchNorm1d(nOut), nn.PReLU())

        if nIn != nOut:
            add = False
        self.add = add

    def forward(self, input):

        output1 = self.c1(input)
        output1 = self.br1(output1)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)

        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8

        combine = torch.cat([d1, add1, add2, add3], 1)

        if self.add:
            combine = input + combine
        output = self.br2(combine)
        return output

class HIL(MessagePassing):#这里继承了父类，所以可以重写父类中的message()函数对他进行覆盖
    def __init__(self, in_channels: int,
                 out_channels: int, 
                 **kwargs):
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


        self.mlp_coord_cov = nn.Sequential(nn.Linear(9, self.in_channels), nn.SiLU())#256
        self.mlp_coord_ncov = nn.Sequential(nn.Linear(9, self.in_channels), nn.SiLU())#256

    def forward(self, x, edge_index_intra, edge_index_inter, pos=None,
                size=None):

        row_cov, col_cov = edge_index_intra
        coord_diff_cov = pos[row_cov] - pos[col_cov]
        radial_cov = self.mlp_coord_cov(_rbf(torch.norm(coord_diff_cov, dim=-1), D_min=0., D_max=6., D_count=9, device=x.device))
        out_node_intra = self.propagate(edge_index=edge_index_intra, x=x, radial=radial_cov, size=size)#message(可自定义将radial融入)

        row_ncov, col_ncov = edge_index_inter
        coord_diff_ncov = pos[row_ncov] - pos[col_ncov]
        radial_ncov = self.mlp_coord_ncov(_rbf(torch.norm(coord_diff_ncov, dim=-1), D_min=0., D_max=6., D_count=9, device=x.device))
        out_node_inter = self.propagate(edge_index=edge_index_inter, x=x, radial=radial_ncov, size=size)

        out_node = self.mlp_node_cov(x + out_node_intra) + self.mlp_node_ncov(x + out_node_inter)

        return out_node

    def message(self, x_j: Tensor, x_i: Tensor, radial, index: Tensor):
        x = x_j * radial #（47988，256）*（47988，256）

        return x


def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design
    
    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count).to(device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF

# out_node_inter = self.propagate(edge_index=edge_index_inter, x=x, radial=radial_ncov, size=size)# %%

class EGNN_complex(nn.Module):
    def __init__(self, hid_dim, edge_dim, n_layers, attention=False, normalize=False, tanh=False):
        super(EGNN_complex, self).__init__()
        self.hid_dim = hid_dim
        self.edge_dim = edge_dim
        self.n_layers = n_layers
        self.attention = attention
        self.normalize = normalize
        self.tanh = tanh

        self.egnn = EGNN(hid_dim, hid_dim, hid_dim, in_edge_nf=edge_dim, n_layers=n_layers, residual=True,
                         attention=attention, normalize=normalize, tanh=tanh)

    def forward(self, data_complex):
        complex_x_list = []
        for i in range(len(data_complex)):
            complex_x = data_complex[i].x
            complex_edge_attr = data_complex[i].edge_attr
            complex_edge_index = data_complex[i].edge_index
            complex_pos = data_complex[i].pos
            if complex_edge_index is None:
                print("Warning: complex_edge_index is None, skipping this edge.")
            complex_x, complex_pos = self.egnn(complex_x, complex_pos, complex_edge_index, complex_edge_attr)
            complex_x_list.append(complex_x)
        complex_x = torch.cat(complex_x_list, dim=0)  # [num_atoms, hid_dim]

        return complex_x

class NEWConvLayer(MessagePassing):
    def __init__(self, input_dim, output_dim, drop=0.1):
        super(NEWConvLayer, self).__init__(aggr='add')  # 使用加法聚合
        self.outmlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Dropout(drop),
            nn.LeakyReLU(),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, x, edge_index, edge_feat):
        # 设置源节点数据和边数据
        out = self.propagate(edge_index=edge_index, x=x, edge_attr = edge_feat)
        out = self.outmlp(out + x)
        return out

    def message(self, x_j: Tensor, x_i: Tensor, edge_attr: Tensor, index: Tensor):
        return F.relu(x_j + edge_attr)
