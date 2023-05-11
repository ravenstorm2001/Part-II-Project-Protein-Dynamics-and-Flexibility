# Pipelines for data extraction

from typing import Literal, Optional
import math
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch_cluster
import torch_geometric

import biotite
import biotite.structure as struc
import biotite.structure.io as strucio
import springcraft
from src.springcraft_NMA import pseudo_fluctuation_measure

STANDARD_AMINO_ACIDS = ["ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU", "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR", "PYL", "SEC" \
                        "XLE", "GLX", "XAA", "ASX"]

_aa_alphabet = {aa: i for i, aa in enumerate(STANDARD_AMINO_ACIDS)}


class DataPreProcessor():
    """
    This class is in the top of the hierarchy. It will be used to load data.
    There are two child classes, depending on the method that is used in the
    training phase.
    """
    def __init__(
        self,
        node_alphabet: dict[str, int],
        edge_cutoff: float = 4.5,
        num_rbf: int = 16,
        self_loop: bool = False,
        device: str = "cpu",
        dtype=torch.float32,
    ):
        self.node_alphabet = node_alphabet
        self.edge_cutoff = edge_cutoff
        self.device = device
        self.dtype = dtype
        self.self_loop = self_loop
        self.num_rbf = num_rbf
        # TODO: Allow various radial functions

    def from_file(self, file_path: str, **kwargs) -> torch_geometric.data.Data:
        """
        Read a protein from a file and return a graph representation.

        Args:
            file_path: str - Path to the file to read.
            **kwargs - Additional arguments to pass to `__call__`.

        Returns:
            torch_geometric.data.Data: Graph representation of the protein.
        """

        whole_structure = strucio.load_structure(file_path)
        protein = whole_structure[struc.filter_amino_acids(whole_structure)]
        # TODO: change from file if necessary to add labels
        label = 0
        return self(protein, label, **kwargs)

    def from_loaded_structure(self, structure: struc.AtomArray, label: int, **kwargs):
        """
        Read a protein from a file and return a graph representation.

        Args:
            structure: struc.AtomArray - protein strucure.
            label: int - class which protein belongs to
            **kwargs -  Additional arguments to pass to `__call__`.

        Returns:
            torch_geometric.data.Data: Graph representation of the protein.
        """
        return self(structure, label, **kwargs)

class DataPreProcessorForGNM(DataPreProcessor):
    """
    This calss is a pre-processor for the core of the project being use of 
    flexibility value that does not depend on the 
    """
    def __init__(
        self,
        node_alphabet: dict[str, int] = _aa_alphabet,
        edge_cutoff: float = 10.0,
        n_pos_embeddings: int = 16,
        type_flexibility: Optional[Literal["msqf", "bfactor", "pseudo"]] = None,
        num_classes: int = 384,
        num_modes: int = 10,
        **kwargs,
    ):
        super().__init__(node_alphabet, edge_cutoff, **kwargs)
        self.n_pos_embeddings = n_pos_embeddings
        self.type_flexibility = type_flexibility
        self.num_classes = num_classes
        self.num_modes = num_modes

    def __call__(self, protein: struc.AtomArray, label: int) -> torch_geometric.data.Data:
        
        # Creating DataFrame, as it is easier to operate on
        df = pd.DataFrame(protein.coord, columns=['x', 'y', 'z'])
        df["chain"] = protein.chain_id
        df["residue"] = protein.res_id
        df["name"] = protein.atom_name
        df["resname"] = protein.res_name
        
        # Get nodes
        coords = torch.as_tensor(self._get_residue_coords(df), device=self.device, dtype=self.dtype)
        residue_type = self._get_residue_types(df)
        help = np.zeros((len(residue_type),len(STANDARD_AMINO_ACIDS)))
        for i in range(len(residue_type)):
            help[i][residue_type[i]] = 1
        node_types = torch.as_tensor(
            help, dtype=torch.long, device=self.device
        )
        mask = torch.isfinite(coords.sum(axis=1))
        coords[~mask] = np.inf

        # Get edges
        edge_index = torch_cluster.radius_graph(coords, r=self.edge_cutoff, loop=self.self_loop)

        # Add node features
        dihedrals = self._dihedrals(df)
        orientations = self._orientations(coords)
        sidechains = self._sidechains(df)
        if self.type_flexibility is None:
            node_s = dihedrals
            node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
        else:
            flexibility = self._flexibility(protein, self.edge_cutoff, self.num_modes)
            node_s = torch.cat([dihedrals, flexibility], dim=-1)
            node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)

        # Add edge features
        pos_embeddings = self._positional_embeddings(edge_index)
        E_vectors = coords[edge_index[0]] - coords[edge_index[1]]
        rbf = _rbf(E_vectors.norm(dim=-1), D_count=self.num_rbf, device=self.device)
        edge_s = torch.cat([rbf, pos_embeddings], dim=-1)
        edge_v = _normalize(E_vectors).unsqueeze(-2)

        # Turn NaN to zeros
        node_s, node_v, edge_s, edge_v = map(torch.nan_to_num, (node_s, node_v, edge_s, edge_v))

        prob = np.zeros(self.num_classes)
        prob[label] = 1

        return torch_geometric.data.Data(
            x=coords,
            y=torch.as_tensor(prob, device=self.device, dtype=self.dtype),
            node_type=node_types,
            edge_index=edge_index,
            node_s=node_s,
            node_v=node_v,
            edge_s=edge_s,
            edge_v=edge_v,
            mask=mask,
        )
        
    def _get_residue_coords(self, df: pd.DataFrame) -> np.ndarray:
        return df[df["name"] == "CA"][["x", "y", "z"]].to_numpy()
    
    def _get_residue_types(self, df: pd.DataFrame) -> np.ndarray:
        return df[df["name"] == "CA"]["resname"].map(self.node_alphabet).to_numpy()

    def _dihedrals(self, df: pd.DataFrame, eps: float = 1e-7) -> torch.Tensor:
        # From https://github.com/jingraham/neurips19-graph-protein-design
        X = torch.as_tensor(
            df[df["name"].isin(["N", "CA", "C", "O"])][["x", "y", "z"]].values.reshape(-1, 4, 3)
        )

        X = torch.reshape(X[:, :3], [3 * X.shape[0], 3])
        dX = X[1:] - X[:-1]
        U = _normalize(dX, dim=-1)
        u_2 = U[:-2]
        u_1 = U[1:-1]
        u_0 = U[2:]

        # Backbone normals
        n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)

        # Angle between normals
        cosD = torch.sum(n_2 * n_1, -1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, [1, 2])
        D = torch.reshape(D, [-1, 3])
        # Lift angle representations to the circle
        D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
        return D_features

    def _positional_embeddings(self, edge_index: torch.Tensor) -> torch.Tensor:
        # From https://github.com/jingraham/neurips19-graph-protein-design
        d = edge_index[0] - edge_index[1]

        frequency = torch.exp(
            torch.arange(0, self.n_pos_embeddings, 2, dtype=self.dtype, device=self.device)
            * -(np.log(10000.0) / self.n_pos_embeddings)
        )
        angles = d.unsqueeze(-1) * frequency
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E

    def _orientations(self, X: torch.Tensor) -> torch.Tensor:
        forward = _normalize(X[1:] - X[:-1])
        backward = _normalize(X[:-1] - X[1:])
        forward = F.pad(forward, [0, 0, 0, 1])
        backward = F.pad(backward, [0, 0, 1, 0])
        return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)

    def _sidechains(self, df: pd.DataFrame) -> torch.Tensor:
        # For each structure, X should be a num_residues x 4 x 3 nested
        # list of the positions of the backbone N, C-alpha, C, and O atoms of
        # each residue (in that order).
        X = torch.as_tensor(
            df[df["name"].isin(["N", "CA", "C", "O"])][["x", "y", "z"]].values.reshape(-1, 4, 3)
        )
        n, origin, c = X[:, 0], X[:, 1], X[:, 2]
        c, n = _normalize(c - origin), _normalize(n - origin)
        bisector = _normalize(c + n)
        perp = _normalize(torch.cross(c, n))
        vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
        return vec
    
    def _flexibility(self, protein: struc.AtomArray, cutoff: float, num_modes: int):
        # Filter CA atoms only
        ca = protein[(protein.atom_name == "CA") & (protein.element == "C")]
    
        # Define force field and GNM object
        ff = springcraft.InvariantForceField(cutoff_distance= cutoff)
        gnm = springcraft.GNM(ca, ff)

        if self.type_flexibility == "pseudo":
            eigenval, eigenvec = gnm.eigen()
            pseudo_fluc = pseudo_fluctuation_measure(eigenval, eigenvec, num_modes, lambda x: 1/x)
            pseudo_fluc = torch.reshape(pseudo_fluc, [-1, 1])
            return pseudo_fluc
        elif self.type_flexibility == "msqf":
            scaler = StandardScaler()
            msqf = torch.from_numpy(gnm.mean_square_fluctuation(mode_subset = np.arange(1, num_modes+1)))
            msqf = torch.reshape(msqf, [-1, 1])
            scaler.fit(msqf)
            return torch.tensor(scaler.transform(msqf))
        else:
            scaler = StandardScaler()
            bfact = torch.from_numpy(gnm.bfactor(mode_subset = np.arange(1, num_modes+1)))
            bfact = torch.reshape(bfact, [-1, 1])
            scaler.fit(bfact)
            return torch.tensor(scaler.transform(bfact)) 

def _normalize(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    """
    return torch.nan_to_num(torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))

def _rbf(
    D: torch.Tensor, D_min: float = 0.0, D_max: float = 20.0, D_count: int = 16, device: str = "cpu"
) -> torch.Tensor:
    """
    From https://github.com/jingraham/neurips19-graph-protein-design

    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    """
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))
    return RBF

def is_chain_valid(chain: biotite.structure.AtomArray):
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
    if ((not all(np.isfinite(ns.coord.sum(axis=1)))) or (not all(np.isfinite(cas.coord.sum(axis=1)))) or (not all(np.isfinite(cs.coord.sum(axis=1)))) or (not all(np.isfinite(os.coord.sum(axis=1))))):
        return False
    if not (set(chain.res_name).issubset(set(STANDARD_AMINO_ACIDS))):
        return False
    if(np.array_equal(ns.res_id, cas.res_id) and np.array_equal(cas.res_id, cs.res_id) and np.array_equal(cs.res_id, os.res_id)):
        return True
    return False

def calculate_all_structures_and_store(labels, database, store_to, num_classes: int = 384, flex: Optional[Literal["msqf", "bfact", "pseudo"]] = None, exclude: Optional[str] = None, num_modes:int = 10):
    if exclude is not None:
        for id in tqdm(set(labels.chain_id).difference(set(line.strip().replace("_",".") for line in open(exclude)))):
            PDBid, chain = id.split(".")
            if(PDBid in database):
                structure = database[PDBid]["structure"]
                x = DataPreProcessorForGNM(type_flexibility = flex, num_modes = num_modes, num_classes=num_classes)
                protein_chain = structure[(structure.chain_id == chain) & struc.filter_amino_acids(structure)] 
                if(is_chain_valid(protein_chain)):
                    if flex is not None:
                        torch.save(x.from_loaded_structure(protein_chain, labels[labels["chain_id"] == id]["label"].iloc[0]), store_to + "/" + flex + "/" + str(labels[labels["chain_id"] == id]["label"].iloc[0]) + "/" + id.replace(".","_") + ".pt")
                    else:
                        torch.save(x.from_loaded_structure(protein_chain, labels[labels["chain_id"] == id]["label"].iloc[0]), store_to + "/no_flex/" + str(labels[labels["chain_id"] == id]["label"].iloc[0]) + "/" + id.replace(".","_") + ".pt")
    else:
        for id in tqdm(set(labels.chain_id)):
            PDBid, chain = id.split(".")
            if(PDBid in database):
                structure = database[PDBid]["structure"]
                x = DataPreProcessorForGNM(type_flexibility = flex, num_modes = num_modes, num_classes=num_classes)
                protein_chain = structure[(structure.chain_id == chain) & struc.filter_amino_acids(structure)] 
                if(is_chain_valid(protein_chain)):
                    if flex is not None:
                        torch.save(x.from_loaded_structure(protein_chain, labels[labels["chain_id"] == id]["label"].iloc[0]), store_to + "/" + flex + "/" + str(labels[labels["chain_id"] == id]["label"].iloc[0]) + "/" + id.replace(".","_") + ".pt")
                    else:
                        torch.save(x.from_loaded_structure(protein_chain, labels[labels["chain_id"] == id]["label"].iloc[0]), store_to + "/no_flex/" + str(labels[labels["chain_id"] == id]["label"].iloc[0]) + "/" + id.replace(".","_") + ".pt")
    
    return