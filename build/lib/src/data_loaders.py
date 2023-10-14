# Classes for wrapping data into datasets

import pandas as pd
import os
import torch
import numpy as np
import biotite
import biotite.structure as struc
import src.data_preprocessing  as dp
from tqdm import tqdm
from typing import Optional, Literal
from torch_geometric.data import Dataset

class ProteinDataset(Dataset):
    """
    Dataset for loading proteins from the paper given splits.
    """
    def __init__(self, labels: pd.DataFrame, include: str, flex: Optional[Literal["msqf", "bfact", "pseudo"]] = "msqf", data_dir: str = "./data/preprocessed_data" ,exclude: Optional[str] = None):
        if flex is None:
            flex = "no_flex"
        self.flex = flex
        self.protein_labels = pd.DataFrame(columns = ['chain_id', 'label'])
        self.data_dir = data_dir
        if exclude != None:
            structure_set = set(line.strip().split(",")[0] for line in open(include)).difference(set(line.strip().replace("_",".") for line in open(exclude)))
        else:
            structure_set = set(line.strip().split(",")[0] for line in open(include))
        for id in tqdm(structure_set):
            if os.path.isfile(self.data_dir + "/" + flex + "/" + str(labels[labels["chain_id"] == id]["label"].iloc[0]) + "/" + str(id).replace(".","_") + ".pt"):
                self.protein_labels = pd.concat([self.protein_labels,  labels[labels["chain_id"] == id]])
        
    def __len__(self):
        return len(self.protein_labels)

    def __getitem__(self, idx):
        label = self.protein_labels.iloc[idx]
        return torch.load(self.data_dir+ "/" + self.flex + "/" + str(label["label"]) + "/" + str(label["chain_id"]).replace(".","_") + ".pt")

def load_labels(path:str):
    """
    Function for loadning labels from the file.

    Arguments:
        path: str - path to labels data
    Return:
        labels: pd.DataFrame - class labels for proteins
    """
    labels = pd.read_csv(path, names = ["chain_id", "label"])
    return labels