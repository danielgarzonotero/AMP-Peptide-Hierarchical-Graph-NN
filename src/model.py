
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, ARMAConv
from torch_geometric.nn import aggr
from torch_scatter import scatter
from src.data import GeoDataset_1
from src.device import device_info
import os

#Hierarchical Graph Neural Network
class GCN_Geo(torch.nn.Module):
    def __init__(self,
                initial_dim_gcn,
                edge_dim_feature,
                hidden_dim_nn_1,
                hidden_dim_nn_2,

                hidden_dim_gat_0,
                
                hidden_dim_fcn_1,
                hidden_dim_fcn_2,
                hidden_dim_fcn_3,
                dropout=0.3):
        super(GCN_Geo, self).__init__()

        self.nn_conv_1 = NNConv(initial_dim_gcn, hidden_dim_nn_1,
                                nn=torch.nn.Sequential(torch.nn.Linear(edge_dim_feature, initial_dim_gcn * hidden_dim_nn_1)), 
                                aggr='add' )
        
        self.nn_conv_2 = NNConv(hidden_dim_nn_1, hidden_dim_nn_2,
                                nn=torch.nn.Sequential(torch.nn.Linear(edge_dim_feature, hidden_dim_nn_1 * hidden_dim_nn_2)), 
                                aggr='add')
        
        #The 7 and 24 comes from the four amino acid features and blosum62 matrix that were concatenated,  95+24
        self.nn_gat_0 = ARMAConv(hidden_dim_nn_2+95, hidden_dim_gat_0, num_stacks = 3, dropout=0.1, num_layers=10, shared_weights = False ) 
        self.readout = aggr.SumAggregation()
        
        #The 7 comes from the four peptides features that were concatenated, +7
        self.linear1 = nn.Linear(hidden_dim_gat_0, hidden_dim_fcn_1)
        self.linear2 = nn.Linear(hidden_dim_fcn_1, hidden_dim_fcn_2 )
        self.linear3 = nn.Linear(hidden_dim_fcn_2, hidden_dim_fcn_3) 
        self.linear4 = nn.Linear(hidden_dim_fcn_3, 1)
        
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self,
                x,
                edge_index,
                edge_attr,
                aminoacids_features_dict,
                blosum62_dict,
                idx_batch,
                cc,
                monomer_labels,
                num_graphs,
                amino): #TODO REVISAR COMO SE GUARDA MONOMER LABEL, Y GUARDAR ASI AMINOACID FEATURES Y SI QUIZAS ASI FUNCIONA LA MASK
        
        x = self.dropout(x)
        x = self.nn_conv_1(x, edge_index, edge_attr)
        x = F.relu(x)
        
        x = self.dropout(x)
        x = self.nn_conv_2(x, edge_index, edge_attr)
        x = F.relu(x)
        
        results_list = []
        
        for i in range(num_graphs):  # Looping over the length of x
            
            mask = idx_batch == i
            
            xi = x[mask]
            monomer_labels_i = monomer_labels[mask]
            cc_i = cc[i].item()
            
            #TODO Intentar remover los diccionarios usando mask
            #monomer_labels_2 = monomer_labels[data.batch == i]
            #xi_2 = x[data.batch == i]
            
            num_aminoacid = torch.max(monomer_labels_i).item()
            amino_index_i = get_amino_indices(num_aminoacid)

            # getting amino acids representation from atom features
            xi = scatter(xi, monomer_labels_i, dim=0, reduce="sum")
            
            # adding amino acids features
            aminoacids_features_i = aminoacids_features_dict[cc_i]
            
            xi = torch.cat((xi, aminoacids_features_i), dim=1)
            #xi = torch.cat((xi, aminoacids_features_i, blosum62_i), dim=1)
            
            # Graph convolution amino acid level
            xi = self.nn_gat_0(xi, amino_index_i) 
            xi = F.relu(xi)
            
            # Readout for peptide representation
            xi = self.readout(xi)
            
            results_list.append(xi)
            
        p = torch.cat(results_list, dim=0)
            
        p = self.linear1(p)
        p = F.relu(p)
        
        p = self.linear2(p)
        p = F.relu(p)
        
        p = self.linear3(p)
        p = F.relu(p)
        
        p = self.linear4(p)
        
        p = torch.transpose(p, 0, 1)
        
        return p


device_info_instance = device_info()
device = device_info_instance.device

def get_amino_indices(num_aminoacid):
    edges = []
    for i in range(num_aminoacid-1):
        edges.append((i, i + 1))
    
    graph_edges = [[x[0] for x in edges], [x[1] for x in edges]]
    
    return torch.tensor(graph_edges, dtype=torch.long, device = device) 


# %%



