import os
import pandas as pd
import numpy as np
import pickle
import multiprocessing
from itertools import repeat
import torch
from torch.utils.data import Dataset, DataLoader
from rdkit import RDLogger
from features_utils import (get_features, inter_graph, drug2emb_encoder,
                            protein2emb_encoder, extract_smiles, read_protein_wo)
from torch_geometric.data import Batch, Data
import warnings

RDLogger.DisableLog('rdApp.*')
np.set_printoptions(threshold=np.inf)
warnings.filterwarnings('ignore')

PT_FEATURE_SIZE = 40
MAX_SMI_LEN = 150

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
