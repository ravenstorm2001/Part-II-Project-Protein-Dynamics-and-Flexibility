# Classes for wrapping data into datasets

import pandas as pd
import os
import torch
import numpy as np
import biotite
import biotite.structure as struc
import src.DataProcessing  as dp
from tqdm import tqdm
from typing import Optional, Literal
from torch_geometric.data import Dataset

class ProteinDataset(Dataset):
    """
    Dataset for loading proteins from the paper given splits.
    """
    def __init__(self, labels, include, flex: Optional[Literal["msqf", "bfact", "pseudo"]] = "msqf", exclude: Optional[str] = None):
        self.flex = flex
        self.protein_labels = pd.DataFrame(columns = ['chain_id', 'label'])
        if exclude != None:
            structure_set = set(line.strip() for line in open(include)).difference(set(line.strip().replace("_",".") for line in open(exclude)))
        else:
            structure_set = set(line.strip() for line in open(include))
        for id in tqdm(structure_set):
            if os.path.isfile("./data/preprocessed_data/" + flex + "/" + str(labels[labels["chain_id"] == id]["label"].iloc[0]) + "/" + str(id).replace(".","_") + ".pt"):
                self.protein_labels = pd.concat([self.protein_labels,  labels[labels["chain_id"] == id]])
        
    def __len__(self):
        return len(self.protein_labels)

    def __getitem__(self, idx):
        label = self.protein_labels.iloc[idx]
        return torch.load("./data/preprocessed_data/" + self.flex + "/" + str(label["label"]) + "/" + str(label["chain_id"]).replace(".","_") + ".pt")

# TODO: REWORK if necessary, or delete if not used till the end of project
class AllProteinDataset(Dataset):
    """
    Dataset class for loading all proteins for available labels.
    """
    def __init__(self, labels, database, flex: Literal["msqf", "bfact", "pseudo"] = "msqf", exclude: Optional[str] = None):
        self.flex = flex
        self.protein_labels = pd.DataFrame(columns = ['chain_id', 'label'])
        for id in tqdm(set(labels.chain_id).difference(set(line.strip().replace("_",".") for line in open(exclude)))):
            PDBid, chain = id.split(".")
            if(PDBid in database):
                structure = database[PDBid]["structure"]
                protein_chain = structure[(structure.chain_id == chain) & struc.filter_amino_acids(structure)] 
                #if(check_chain(protein_chain)):
                #    self.protein_labels = pd.concat([self.protein_labels,labels[labels["chain_id"] == id]])
    
    def __len__(self):
        return len(self.protein_labels)

    def __getitem__(self, idx):
        # TODO: fixERROR if used (iloc only gets one value)
        labeled = self.protein_labels.iloc[idx]
        PDBid, chain = labeled.iloc[0]["chain_id"].split(".")
        x = dp.DataPreProcessorForGNM(type_flexibility = self.flex)
        structure = self.database[PDBid]["structure"]
        protein_chain = structure[(structure.chain_id == chain) & struc.filter_amino_acids(structure)] 
        struct = [x.from_loaded_structure(protein_chain, m) for m in labeled["label"]]
        return struct


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