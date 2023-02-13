# Classes for wrapping data into datasets

import pandas as pd
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
    def __init__(self, labels, database, include, flex: Literal["msqf", "bfact", "pseudo"] = "msqf", exclude: Optional[str] = None):
        self.flex = flex
        self.database = database
        self.structures = []
        self.protein_labels = pd.DataFrame(columns = ['chain_id', 'label'])
        if exclude != None:
            structure_set = set(line.strip() for line in open(include)).difference(set(line.strip().replace("_",".") for line in open(exclude)))
        else:
            structure_set = set(line.strip() for line in open(include))
        for id in tqdm(structure_set):
            PDBid, chain = id.split(".")
            if(PDBid in database):
                structure = database[PDBid]["structure"]
                protein_chain = structure[(structure.chain_id == chain) & struc.filter_amino_acids(structure)] 
                if(check_chain(protein_chain)):
                    self.protein_labels = pd.concat([self.protein_labels,  labels[labels["chain_id"] == id]])
        
    def __len__(self):
        return len(self.protein_labels)

    def __getitem__(self, idx):
        labeled = self.protein_labels.iloc[idx]
        PDBid, chain = labeled.iloc[0]["chain_id"].split(".")
        x = dp.DataPreProcessorForGNM(type_flexibility = self.flex)
        structure = self.database[PDBid]["structure"]
        protein_chain = structure[(structure.chain_id == chain) & struc.filter_amino_acids(structure)] 
        struct = [x.from_loaded_structure(protein_chain, m) for m in labeled["label"]]
        return struct

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
                if(check_chain(protein_chain)):
                    self.protein_labels = pd.concat([self.protein_labels,labels[labels["chain_id"] == id]])
    
    def __len__(self):
        return len(self.protein_labels)

    def __getitem__(self, idx):
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

def check_chain(chain: biotite.structure.AtomArray):
    """
    Filtering function. Checks whether we have same number of N,CA,C,O atoms and whether they are present for the same subset of residues.

    Arguments:
        chain: biotite.structure.AtomArray - protein chain to check
    Return:
        result: bool - whether the atom is acceptible
    """
    ns = chain[chain.atom_name == 'N']
    cas = chain[chain.atom_name == 'CA']
    cs = chain[chain.atom_name == 'C']
    os = chain[chain.atom_name == 'O']
    if not (set(chain.res_name).issubset(set(dp.STANDARD_AMINO_ACIDS))):
        return False
    if(np.array_equal(ns.res_id, cas.res_id) and np.array_equal(cas.res_id, cs.res_id) and np.array_equal(cs.res_id, os.res_id)):
        return True
    return False