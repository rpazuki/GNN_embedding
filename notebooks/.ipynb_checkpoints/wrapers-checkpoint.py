import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, GATConv, GCNConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric import data as DATA
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import IterableDataset
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
import datamol as dm

# GINConv model
class GINConvNet_augmented(torch.nn.Module):
    def __init__(self, n_output=1,num_features_xd=78, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):

        super(GINConvNet_augmented, self).__init__()

        dim = 32
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output
        # convolution layers
        nn1 = Sequential(Linear(num_features_xd, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1_xd = Linear(dim, output_dim)

        # 1D convolution on protein sequence
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        self.fc1_xt = nn.Linear(32*121, output_dim)

        # combined layers
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, self.n_output)        # n_output = 1 for regression task

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        target = data.target

        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1_xd(x))
        x = F.dropout(x, p=0.2, training=self.training)

        embedded_xt = self.embedding_xt(target)
        conv_xt = self.conv_xt_1(embedded_xt)
        # flatten
        xt = conv_xt.view(-1, 32 * 121)
        xt = self.fc1_xt(xt)

        # concat
        xc = torch.cat((x, xt), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out
        
    def forward_dummy(self, data):
        x, edge_index, batch = data
        

        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1_xd(x))
        x = F.dropout(x, p=0.2, training=self.training)

        return x
        
    def g_embedding(self, data):
        x, edge_index, id = data
        # We assumed a batch with only one elements has been provided.
        # So, for edge_index with (none, V,V), we squeeze the batch dimension
        #print(x.shape, edge_index.shape)
        edge_index = torch.squeeze(edge_index, 0)
        x = torch.squeeze(x, 0)
        #print(x.shape, edge_index.shape)
        #batch = data.batch                
        x = F.relu(self.conv1(x, edge_index))        
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        #x = global_add_pool(x, batch)
        x = global_add_pool(x, batch=None)
        x = self.fc1_xd(x)
        x = F.dropout(x, p=0.2, training=self.training)
                
        return x, np.array(id, dtype='<U7')

class GATNet_augmented(torch.nn.Module):
    def __init__(self, num_features_xd=78, n_output=1, num_features_xt=25,
                     n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):
        super(GATNet_augmented, self).__init__()

        # graph layers
        self.gcn1 = GATConv(num_features_xd, num_features_xd, heads=10, dropout=dropout)
        self.gcn2 = GATConv(num_features_xd * 10, output_dim, dropout=dropout)
        self.fc_g1 = nn.Linear(output_dim, output_dim)

        # 1D convolution on protein sequence
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        self.fc_xt1 = nn.Linear(32*121, output_dim)

        # combined layers
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, n_output)

        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        # graph input feed-forward
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.gcn1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.gcn2(x, edge_index)
        x = self.relu(x)
        x = gmp(x, batch)          # global max pooling
        x = self.fc_g1(x)
        x = self.relu(x)

        # protein input feed-forward:
        target = data.target
        embedded_xt = self.embedding_xt(target)
        conv_xt = self.conv_xt1(embedded_xt)
        conv_xt = self.relu(conv_xt)

        # flatten
        xt = conv_xt.view(-1, 32 * 121)
        xt = self.fc_xt1(xt)

        # concat
        xc = torch.cat((x, xt), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out   

    def g_embedding(self, data):
        x, edge_index, id = data
        # We assumed a batch with only one elements has been provided.
        # So, for edge_index with (none, V,V), we squeeze the batch dimension
        edge_index = torch.squeeze(edge_index, 0)
        x = torch.squeeze(x, 0)
        #
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.gcn1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.gcn2(x, edge_index)
        x = self.relu(x)
        x = gmp(x, batch=None)          # global max pooling
        x = self.fc_g1(x)
        x = self.relu(x)

        return x, np.array(id, dtype='<U7')

class GAT_GCN_augmented(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=78, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):

        super(GAT_GCN_augmented, self).__init__()

        self.n_output = n_output
        self.conv1 = GATConv(num_features_xd, num_features_xd, heads=10)
        self.conv2 = GCNConv(num_features_xd*10, num_features_xd*10)
        self.fc_g1 = torch.nn.Linear(num_features_xd*10*2, 1500)
        self.fc_g2 = torch.nn.Linear(1500, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # 1D convolution on protein sequence
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        self.fc1_xt = nn.Linear(32*121, output_dim)

        # combined layers
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)        # n_output = 1 for regression task

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        target = data.target
        # print('x shape = ', x.shape)
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        # apply global max pooling (gmp) and global mean pooling (gap)
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)

        embedded_xt = self.embedding_xt(target)
        conv_xt = self.conv_xt_1(embedded_xt)
        # flatten
        xt = conv_xt.view(-1, 32 * 121)
        xt = self.fc1_xt(xt)

        # concat
        xc = torch.cat((x, xt), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out

    def g_embedding(self, data):
        x, edge_index, id = data
        # We assumed a batch with only one elements has been provided.
        # So, for edge_index with (none, V,V), we squeeze the batch dimension
        edge_index = torch.squeeze(edge_index, 0)
        x = torch.squeeze(x, 0)
        #        
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        # apply global max pooling (gmp) and global mean pooling (gap)
        x = torch.cat([gmp(x, batch=None), gap(x, batch=None)], dim=1)
        x = self.relu(self.fc_g1(x))
        x = self.dropout(x)
        x = self.fc_g2(x)

        return x, np.array(id, dtype='<U7')



# From GraphDTA lib
def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))
def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    
    c_size = mol.GetNumAtoms()
    
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append( feature / sum(feature) )

    edges = []    
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    # Roozbeh
    # For single atoms or ions, I created a self-loop
    if len(edges) == 0:
        for atom in mol.GetAtoms():
            edges.append([atom.GetIdx(), atom.GetIdx()])
            
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
        
    return c_size, features, edge_index

###################################

def smiles_to_graphs(ids, smiles, isomericSmiles=True, use_rdkit = True):  
    smiles_ids = []
    for i, (id,s) in enumerate(zip(ids,smiles)):
        try:
            if use_rdkit:
                m = Chem.MolFromSmiles(s)
            else:
                mol = dm.to_mol(s, ordered=True)
                mol = dm.fix_mol(mol)
                mol = dm.sanitize_mol(mol, sanifix=True, charge_neutral=False)
                m = dm.standardize_mol(
                    mol,
                    disconnect_metals=False,
                    normalize=True,
                    reionize=True,
                    uncharge=False,
                    stereo=True,
                )            
            if m is None:
                m = dm.to_mol(s)
                if m is None:
                    print(f"The smiles id {i}:{id} has been failed.")
                    continue
            lg = Chem.MolToSmiles(m,isomericSmiles=isomericSmiles)
            #compound_iso_smiles.append(lg)
            #smiles_id_dict[lg] = id
            smiles_ids.append((lg, id))
        except Exception as e:
            print(f"The smiles id {i}:{id} has been failed.", e)
    #compound_iso_smiles = list(set(compound_iso_smiles))
    smile_graph = {}
    for smile, _ in smiles_ids:
        g = smile_to_graph(smile)
        smile_graph[smile] = g
        if len(g) == 0:
            print(smiles, g)

    compound_iso_smiles = []
    smiles_id_dict = {}
    for smile, id in smiles_ids:
        compound_iso_smiles.append(smile)
        smiles_id_dict[smile] = id
    return compound_iso_smiles, smiles_id_dict, smile_graph

class CompoundsStream(IterableDataset):

    def __init__(self, smiles_list, smiles_graphs, smiles_id_dict):
        super().__init__()
        self.smiles_list = smiles_list
        self.smiles_id_dict = smiles_id_dict
        self.smiles_graphs = smiles_graphs

    def generate(self):
        i = 0
        for smiles in self.smiles_list:             
            i += 1
            id = self.smiles_id_dict[smiles]
            # convert SMILES to molecular representation using rdkit
            c_size, features, edge_index = self.smiles_graphs[smiles]
            if len(edge_index) == 0:
                
                print(f"Empty edge_index for id:{id}, smiles:{smiles}, edge_index:{edge_index}")
                continue
                         
            #yield GCNData
            yield torch.Tensor(features), torch.LongTensor(np.array(edge_index)).transpose(1, 0), id

    def __iter__(self):
        return iter(self.generate())

class CompoundsStreamFull:

    def __init__(self, smiles_list, smiles_graphs, smiles_id_dict):        
        self.smiles_list = smiles_list
        self.smiles_id_dict = smiles_id_dict
        self.smiles_graphs = smiles_graphs
        
        self.features = []
        self.edge_indices = []
        self.batch = []
        i = 0
        for smiles in self.smiles_list:             
            i += 1
            id = self.smiles_id_dict[smiles]
            # convert SMILES to molecular representation using rdkit
            c_size, features, edge_index = self.smiles_graphs[smiles]
            if len(edge_index) == 0:
                
                print(f"Empty edge_index for id:{id}, smiles:{smiles}, edge_index:{edge_index}")
                continue
                         
            self.features.append(np.array(features))
            self.edge_indices.append(np.array(edge_index))
            self.batch += [i] * len(features)
                                     

    def get_data(self):
        return (torch.tensor(np.vstack([*self.features]), dtype=torch.float32), 
                torch.tensor(np.vstack([*self.edge_indices]).T),
                torch.tensor(np.array(self.batch)))


    
