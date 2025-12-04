import os
import pandas as pd
import numpy as np
import pickle
import multiprocessing
import torch
from torch.utils.data import Dataset, DataLoader
from rdkit import RDLogger
from rdkit import Chem
import codecs
import networkx as nx
from scipy.spatial import distance_matrix
from subword_nmt.apply_bpe import BPE
import re
from utils import cal_dist, area_triangle, angle, PT_FEATURE_SIZE
from torch_geometric.data import Batch, Data
import warnings

RDLogger.DisableLog('rdApp.*')
np.set_printoptions(threshold=np.inf)
warnings.filterwarnings('ignore')

# use PT_FEATURE_SIZE from utils

# Function to create paths for all components of a sample
def generate_paths(data_dir, cid, dis_threshold, seq_type='p_lm'):
    complex_dir = os.path.join(data_dir, cid)
    return {
        "graph_path": os.path.join(complex_dir, f"Graph_DualBind-{cid}-{dis_threshold}.pyg"),
        "complex_path": os.path.join(complex_dir, f"{cid}_{dis_threshold}A.rdkit"),
        "pocket_path": os.path.join(data_dir, 'pocket', f"{cid}.csv"),
        "prot_path": os.path.join(complex_dir, f"{cid}_protein.pdb"),
        "seq_path": os.path.join(complex_dir, f"{seq_type}_{cid}.pt")
    }


# ===== Inlined from features_utils =====
CHAR_SMI_SET = {"(": 1, ".": 2, "0": 3, "2": 4, "4": 5, "6": 6, "8": 7, "@": 8,
                "B": 9, "D": 10, "F": 11, "H": 12, "L": 13, "N": 14, "P": 15, "R": 16,
                "T": 17, "V": 18, "Z": 19, "\\": 20, "b": 21, "d": 22, "f": 23, "h": 24,
                "l": 25, "n": 26, "r": 27, "t": 28, "#": 29, "%": 30, ")": 31, "+": 32,
                "-": 33, "/": 34, "1": 35, "3": 36, "5": 37, "7": 38, "9": 39, "=": 40,
                "A": 41, "C": 42, "E": 43, "G": 44, "I": 45, "K": 46, "M": 47, "O": 48,
                "S": 49, "U": 50, "W": 51, "Y": 52, "[": 53, "]": 54, "a": 55, "c": 56,
                "e": 57, "g": 58, "i": 59, "m": 60, "o": 61, "s": 62, "u": 63, "y": 64}

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(mol, graph, atom_symbols=['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I'], explicit_H=True):
    for atom in mol.GetAtoms():
        results = one_of_k_encoding_unk(atom.GetSymbol(), atom_symbols + ['Unknown']) + \
                one_of_k_encoding_unk(atom.GetDegree(),[0, 1, 2, 3, 4, 5, 6]) + \
                one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
                one_of_k_encoding_unk(atom.GetHybridization(), [
                    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                        SP3D, Chem.rdchem.HybridizationType.SP3D2
                    ]) + [atom.GetIsAromatic()]
        if explicit_H:
            results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
        graph.add_node(atom.GetIdx(), feats=torch.from_numpy(np.array(results).astype(np.float32)))

def get_edge_index(mol, graph):
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        graph.add_edge(i, j)

def edge_features(mol, graph):
    geom = mol.GetConformers()[0].GetPositions()
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        angles_ijk, areas_ijk, dists_ik = [], [], []
        for neighbor in mol.GetAtomWithIdx(j).GetNeighbors():
            k = neighbor.GetIdx()
            if mol.GetBondBetweenAtoms(j, k) is not None and i != k:
                vector1 = geom[j] - geom[i]
                vector2 = geom[k] - geom[i]
                angles_ijk.append(angle(vector1, vector2))
                areas_ijk.append(area_triangle(vector1, vector2))
                dists_ik.append(cal_dist(geom[i], geom[k]))
        angles_ijk = np.array(angles_ijk) if angles_ijk != [] else np.array([0.])
        areas_ijk = np.array(areas_ijk) if areas_ijk != [] else np.array([0.])
        dists_ik = np.array(dists_ik) if dists_ik != [] else np.array([0.])
        dist_ij1 = cal_dist(geom[i], geom[j], ord=1)
        dist_ij2 = cal_dist(geom[i], geom[j], ord=2)
        geom_feats = [
            angles_ijk.max()*0.1,
            angles_ijk.sum()*0.01,
            angles_ijk.mean()*0.1,
            areas_ijk.max()*0.1,
            areas_ijk.sum()*0.01,
            areas_ijk.mean()*0.1,
            dists_ik.max()*0.1,
            dists_ik.sum()*0.01,
            dists_ik.mean()*0.1,
            dist_ij1*0.1,
            dist_ij2*0.1,
        ]
        bond_type = bond.GetBondType()
        basic_feats = [
            bond_type == Chem.rdchem.BondType.SINGLE,
            bond_type == Chem.rdchem.BondType.DOUBLE,
            bond_type == Chem.rdchem.BondType.TRIPLE,
            bond_type == Chem.rdchem.BondType.AROMATIC,
            bond.GetIsConjugated(),
            bond.IsInRing()]
        graph.add_edge(i, j, feats=torch.tensor(basic_feats + geom_feats).float())

def get_features(mol):
    graph = nx.Graph()
    atom_features(mol, graph)
    get_edge_index(mol, graph)
    edge_features(mol, graph)
    graph = graph.to_directed()
    atom_feature = torch.stack([feats['feats'] for n, feats in graph.nodes(data=True)])
    edge_index = torch.stack([torch.LongTensor((u, v)) for u, v in graph.edges(data=False)]).T
    edge_feature = torch.stack([feats['feats'] for u, v, feats in graph.edges(data=True)])
    return atom_feature, edge_index, edge_feature

def geom_feat(pos_i, pos_j, pos_k, angles_ijk, areas_ijk, dists_ik):
    vector1 = pos_j - pos_i
    vector2 = pos_k - pos_i
    angles_ijk.append(angle(vector1, vector2))
    areas_ijk.append(area_triangle(vector1, vector2))
    dists_ik.append(cal_dist(pos_i, pos_k))

def geom_feats(pos_i, pos_j, angles_ijk, areas_ijk, dists_ik):
    angles_ijk = np.array(angles_ijk) if angles_ijk != [] else np.array([0.])
    areas_ijk = np.array(areas_ijk) if areas_ijk != [] else np.array([0.])
    dists_ik = np.array(dists_ik) if dists_ik != [] else np.array([0.])
    dist_ij1 = cal_dist(pos_i, pos_j, ord=1)
    dist_ij2 = cal_dist(pos_i, pos_j, ord=2)
    geom = [
        angles_ijk.max()*0.1,
        angles_ijk.sum()*0.01,
        angles_ijk.mean()*0.1,
        areas_ijk.max()*0.1,
        areas_ijk.sum()*0.01,
        areas_ijk.mean()*0.1,
        dists_ik.max()*0.1,
        dists_ik.sum()*0.01,
        dists_ik.mean()*0.1,
        dist_ij1*0.1,
        dist_ij2*0.1,
    ]
    return geom

def inter_graph(ligand, pocket, dis_threshold = 5.):
    atom_num_l = ligand.GetNumAtoms()
    atom_num_p = pocket.GetNumAtoms()
    graph_inter = nx.Graph()
    pos_l = ligand.GetConformers()[0].GetPositions()
    pos_p = pocket.GetConformers()[0].GetPositions()
    dis_matrix = distance_matrix(pos_l, pos_p)
    node_idx = np.where(dis_matrix < dis_threshold)
    for i, j in zip(node_idx[0], node_idx[1]):
        ks = node_idx[0][node_idx[1] == j]
        angles_ijk, areas_ijk, dists_ik = [], [], []
        for k in ks:
            if k != i:
                geom_feat(pos_l[i], pos_p[j], pos_l[k], angles_ijk, areas_ijk, dists_ik)
        geom = geom_feats(pos_l[i], pos_p[j], angles_ijk, areas_ijk, dists_ik)
        bond_feats = torch.FloatTensor(geom)
        graph_inter.add_edge(i, j + atom_num_l, feats=torch.tensor(bond_feats))
    graph_inter = graph_inter.to_directed()
    edge_index_inter = torch.stack([torch.LongTensor((u, v)) for u, v in graph_inter.edges(data=False)]).T
    edge_feats_inter = torch.stack([feats['feats'] for _, _, feats in graph_inter.edges(data=True)]).float()
    return edge_index_inter, edge_feats_inter

def drug2emb_encoder(x):
    vocab_path = './ESPF/drug_codes_chembl.txt'
    bpe_codes_drug = codecs.open(vocab_path)
    dbpe = BPE(bpe_codes_drug, merges=-1, separator='')
    sub_csv = pd.read_csv('./ESPF/subword_units_map_chembl.csv')
    idx2word_d = sub_csv['index'].values
    words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))
    max_d = 50
    t1 = dbpe.process_line(x).split()
    try:
        i1 = np.asarray([words2idx_d[i] for i in t1])
    except:
        i1 = np.array([0])
    l = len(i1)
    if l < max_d:
        i = np.pad(i1, (0, max_d - l), 'constant', constant_values=0)
        input_mask = ([1] * l) + ([0] * (max_d - l))
    else:
        i = i1[:max_d]
        input_mask = [1] * max_d
    return i, np.asarray(input_mask)

def protein2emb_encoder(x):
    vocab_path = './ESPF/protein_codes_uniprot.txt'
    bpe_codes_protein = codecs.open(vocab_path)
    pbpe = BPE(bpe_codes_protein, merges=-1, separator='')
    sub_csv = pd.read_csv('./ESPF/subword_units_map_uniprot.csv')
    idx2word_p = sub_csv['index'].values
    words2idx_p = dict(zip(idx2word_p, range(0, len(idx2word_p))))
    max_p = 256
    t1 = pbpe.process_line(x).split()
    try:
        i1 = np.asarray([words2idx_p[i] for i in t1])
    except:
        i1 = np.array([0])
    l = len(i1)
    if l < max_p:
        i = np.pad(i1, (0, max_p - l), 'constant', constant_values=0)
        input_mask = ([1] * l) + ([0] * (max_p - l))
    else:
        i = i1[:max_p]
        input_mask = [1] * max_p
    return i, np.asarray(input_mask)

def extract_smiles(mol):
    if mol is None:
        raise ValueError("Molecule is None.")
    smiles = Chem.MolToSmiles(mol)
    return smiles

def read_protein_wo(filepath):
    aa_codes = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
        'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'LYS': 'K',
        'ILE': 'I', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
        'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
        'THR': 'T', 'VAL': 'V', 'TYR': 'Y', 'TRP': 'W'}
    seq = ''
    for line in open(filepath):
        if line[0:6] == "SEQRES":
            columns = line.split()
            for resname in columns[4:]:
                if resname in aa_codes:
                    seq = seq + aa_codes[resname]
    return seq

# ===== End of inlined functions =====


def mols2graphs(cid, paths, label, dis_threshold):
    try:
        with open(paths['complex_path'], 'rb') as f:
            ligand, pocket = pickle.load(f)

        # Process ligand and pocket features
        x_l, edge_index_l, edge_feature_l = get_features(ligand)
        x_p, edge_index_p, edge_feature_p = get_features(pocket)
        atom_num_l, atom_num_p = ligand.GetNumAtoms(), pocket.GetNumAtoms()

        x = torch.cat([x_l, x_p], dim=0)
        edge_index_intra = torch.cat([edge_index_l, edge_index_p + atom_num_l], dim=-1)
        edge_index_inter, edge_feats_inter = inter_graph(ligand, pocket, dis_threshold=dis_threshold)
        pos = torch.cat([torch.FloatTensor(ligand.GetConformers()[0].GetPositions()),
                         torch.FloatTensor(pocket.GetConformers()[0].GetPositions())], dim=0)
        split = torch.cat([torch.zeros(atom_num_l), torch.ones(atom_num_p)], dim=0)

        # Create data object
        data_complex = Data(
            x=x, edge_index_intra=edge_index_intra, edge_index_inter=edge_index_inter,
            y=torch.FloatTensor([label]), pos=pos, split=split
        )

        # Extract sequence features
        smiles = extract_smiles(ligand)
        prot_seq = read_protein_wo(paths['prot_path'])
        lig_trans, lig_trans_mask = drug2emb_encoder(smiles)
        pro_trans, pro_trans_mask = protein2emb_encoder(prot_seq)

        seq_features = {
            'pro_trans': pro_trans, 'pro_mask': pro_trans_mask,
            'smi_trans': lig_trans, 'smi_mask': lig_trans_mask
        }

        # Save processed data
        torch.save(data_complex, paths['graph_path'])
        torch.save(seq_features, paths['seq_path'])

    except Exception as e:
        print(f"Error processing {cid}: {e}")


class GraphDataset(Dataset):
    def __init__(self, data_dir, data_df, dis_threshold=5, num_process=4, create=True):
        self.data_dir = data_dir
        self.data_df = data_df
        self.dis_threshold = dis_threshold
        self.create = create
        self.paths = [
            (row['pdbid'], generate_paths(data_dir, row['pdbid'], dis_threshold), row['-logKd/Ki'])
            for _, row in data_df.iterrows()
        ]

        if self.create:
            print('Generating data...')
            with multiprocessing.Pool(num_process) as pool:
                pool.starmap(
                    mols2graphs,
                    [(cid, paths, label, dis_threshold) for cid, paths, label in self.paths]
                )

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        cid, paths, _ = self.paths[idx]
        data_complex = torch.load(paths['graph_path'])
        seq_features = torch.load(paths['seq_path'])
        pkt_tensor = pd.read_csv(paths['pocket_path']).drop(columns=['idx'], errors='ignore').values[:60]

        pkt_tensor_padded = np.zeros((60, PT_FEATURE_SIZE), dtype=np.float32)
        pkt_tensor_padded[:len(pkt_tensor)] = pkt_tensor

        return data_complex, seq_features, pkt_tensor_padded

    def collate_fn(self, batch):
        complex, sequence, pkt_tensor = zip(*batch)

        batch_graphs = Batch.from_data_list(complex)
        batch_pkt = torch.from_numpy(np.stack(pkt_tensor, axis=0).astype(np.float32))

        batch_prot_trans = torch.stack([torch.from_numpy(item['pro_trans']) for item in sequence], dim=0)
        batch_mask_p = torch.stack([torch.from_numpy(item['pro_mask']) for item in sequence], dim=0)
        batch_lig_trans = torch.stack([torch.from_numpy(item['smi_trans']) for item in sequence], dim=0)
        batch_mask_l = torch.stack([torch.from_numpy(item['smi_mask']) for item in sequence], dim=0)

        return {
            'complex': batch_graphs, 'pkt_tensor': batch_pkt,
            'prot_trans': batch_prot_trans, 'prot_msk': batch_mask_p,
            'smi_trans': batch_lig_trans, 'smi_msk': batch_mask_l
        }


class PLIDataLoader(DataLoader):
    def __init__(self, data, **kwargs):
        super().__init__(data, collate_fn=data.collate_fn, **kwargs)


if __name__ == '__main__':
    data_root = './data'
    toy_df = pd.read_csv(os.path.join(data_root, "toyset.csv"))
    toy_set = GraphDataset(
        data_dir=os.path.join(data_root, 'toyset'),
        data_df=toy_df,
        dis_threshold=5,
        create=True
    )
    train_loader = PLIDataLoader(toy_set, batch_size=128, shuffle=True, num_workers=4)
