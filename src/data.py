import pandas as pd
import os
import torch
from torch_geometric.data import InMemoryDataset

from src.utils import sequences_geodata, get_features
from src.device import device_info
from src.aminoacids_features import get_aminoacid_features

class GeoDatasetBase(InMemoryDataset):
    def __init__(self, root='', raw_name='', transform=None, pre_transform=None, **kwargs):
        self.filename = raw_name  # La ruta completa ya se proporciona
        self.df = pd.read_csv(self.filename)
        self.x = self.df[self.df.columns[0]].values
        self.y = self.df[self.df.columns[1:6]].values  
        super(GeoDatasetBase, self).__init__(root=os.path.join(root, f'{raw_name.split(".")[0]}_processed'), transform=transform, pre_transform=pre_transform, **kwargs)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def process(self):
        node_ft_dict, edge_ft_dict = get_features(self.x)
        data_list = []
        cc = 0
        aminoacids_ft_dict = get_aminoacid_features()
        
        for i, (x, y) in enumerate(zip(self.x, self.y)):
            device_info_instance = device_info()
            device = device_info_instance.device
            data_list.append(sequences_geodata(cc, x, y, aminoacids_ft_dict, node_ft_dict, edge_ft_dict, device))
            cc += 1
            
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class GeoDataset(GeoDatasetBase):
    def __init__(self, root='', raw_name='', transform=None, pre_transform=None, **kwargs):
        super(GeoDataset, self).__init__(root=root, raw_name=raw_name, transform=transform, pre_transform=pre_transform, **kwargs)

    def processed_file_names(self):
        return [f'{os.path.splitext(os.path.basename(self.filename))[0]}.pt']
