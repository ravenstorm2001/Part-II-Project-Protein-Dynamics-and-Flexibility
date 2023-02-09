import pandas as pd
import numpy as np
import biotite
import biotite.structure as struc
import src.DataProcessing  as dp
from tqdm import tqdm
from typing import Optional

def load_labels(path):
    labels = pd.read_csv(path, names = ["chain_id", "label"])
    labels[['id', 'chain']] = labels['chain_id'].str.split('.', 1, expand=True)
    return labels

def check_chain(chain: biotite.structure.AtomArray):
    ns = chain[chain.atom_name == 'N']
    cas = chain[chain.atom_name == 'CA']
    cs = chain[chain.atom_name == 'C']
    os = chain[chain.atom_name == 'O']
    if not (set(chain.res_name).issubset(set(dp.STANDARD_AMINO_ACIDS))):
        return False
    if(np.array_equal(ns.res_id, cas.res_id) and np.array_equal(cas.res_id, cs.res_id) and np.array_equal(cs.res_id, os.res_id)):
        return True
    return False

def load_structures_from_file_ids(path, labels, database, exclude: Optional[str] = None):
    if exclude != None:
        structure_set = set(line.strip() for line in open(path)).difference(set(line.strip().replace("_",".") for line in open(exclude)))
    else:
        structure_set = set(line.strip() for line in open(path))
    structures = []
    for id in tqdm(structure_set):
        PDBid, chain = id.split(".")
        x = dp.DataPreProcessorForGNM(type_flexibility = "msqf")
        if(database.__contains__(PDBid)):
            structure = database[PDBid]["structure"]
            protein_chain = structure[(structure.chain_id == chain) & struc.filter_amino_acids(structure)] 
            if(check_chain(protein_chain)):
                structures.append(x.from_loaded_structure(protein_chain, labels[(labels["id"] == PDBid) & (labels["chain"] == chain)].label.iloc[0]))
    return structures

def load_structures_from_labels_available(labels, database, exclude):
    all_structures = []
    for id in tqdm(set(labels.chain_id).difference(set(line.strip().replace("_",".") for line in open('./data/enzyme_data/missing_resis.txt')))):
        PDBid, chain = id.split(".")
        x = dp.DataPreProcessorForGNM(type_flexibility = "msqf")
        if(database.__contains__(PDBid)):
            structure = database[PDBid]["structure"]
            protein_chain = structure[(structure.chain_id == chain) & struc.filter_amino_acids(structure)] 
            if(check_chain(protein_chain)):
                all_structures.append(x.from_loaded_structure(protein_chain, labels[labels.chain_id == id].label.iloc[0]))