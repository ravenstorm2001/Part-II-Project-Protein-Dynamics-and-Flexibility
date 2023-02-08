# This file is used to calculate/extract and plot normal modes from a file or from PDBID
import biotite
import biotite.structure as struc
import biotite.structure.io.mmtf as mmtf
import biotite.database.rcsb as rcsb
import biotite.structure.io as strucio
import springcraft
import scipy
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Literal

def calculate_normal_modes_gnm_from_file(path, cutoff):
    '''
    Function that calculates eigenvalues and eigenvectors of Kirchhoff matrix using GNM method from .pdb file.

    Arguments:
        path : string - Path to .pdb file containing structure we want to explore
        cutoff : double - Angstrom distance under which two CA atoms are represented as connected
    Returns:
        gnm : springcraft.GNM - class that contains Kirchhoff matrix and can call calculation of normal modes
    '''

    # Read .pdb file and filter CA atoms
    whole_structure = strucio.load_structure(path)
    protein = whole_structure[struc.filter_amino_acids(whole_structure)]
    ca = protein[(protein.atom_name == "CA") & (protein.element == "C")]
    
    # Define force field and GNM object
    ff = springcraft.InvariantForceField(cutoff_distance=cutoff)
    gnm = springcraft.GNM(ca, ff)

    # Return model with all pre-processing
    return gnm

def calculate_normal_modes_gnm_from_id(id, cutoff):
    '''
    Function that calculates eigenvalues and eigenvectors of Kirchhoff matrix using GNM method from PDBID of a structure.

    Arguments:
        id : string - PDBID of a structure we want to explore
        cutoff : double - Angstrom distance under which two CA atoms are represented as connected
    Returns:
        gnm : springcraft.GNM - class that contains Kirchhoff matrix and can call calculation of normal modes
    '''

    # Read .pdb file and filter CA atoms
    mmtf_file = mmtf.MMTFFile.read(rcsb.fetch(id, "mmtf"))
    whole_structure = mmtf.get_structure(mmtf_file, model=1, include_bonds=True)
    protein = whole_structure[struc.filter_amino_acids(whole_structure)]
    ca = protein[(protein.atom_name == "CA") & (protein.element == "C")]
    
    # Define force field and GNM object
    ff = springcraft.InvariantForceField(cutoff_distance=cutoff)
    gnm = springcraft.GNM(ca, ff)

    # Return model with all pre-processing
    return gnm    

def calculate_normal_modes_anm_from_file(path, cutoff):
    '''
    Function that calculates eigenvalues and eigenvectors of Kirchhoff matrix using ANM method from .pdb file.

    Arguments:
        path : string - Path to .pdb file containing structure we want to explore
        cutoff : double - Angstrom distance under which two CA atoms are represented as connected
    Returns:
        anm : springcraft.GNM - class that contains Kirchhoff matrix and can call calculation of normal modes
    '''

    # Read .pdb file and filter CA atoms
    whole_structure = strucio.load_structure(path)
    protein = whole_structure[struc.filter_amino_acids(whole_structure)]
    ca = protein[(protein.atom_name == "CA") & (protein.element == "C")]
    
    # Define force field and GNM object
    ff = springcraft.InvariantForceField(cutoff_distance=cutoff)
    anm = springcraft.ANM(ca, ff)

    # Return model with all pre-processing
    return anm

def calculate_normal_modes_anm_from_id(id, cutoff):
    '''
    Function that calculates eigenvalues and eigenvectors of Kirchhoff matrix using ANM method from PDBID of a structure.

    Arguments:
        id : string - PDBID of a structure we want to explore
        cutoff : double - Angstrom distance under which two CA atoms are represented as connected
    Returns:
        anm : springcraft.GNM - class that contains Kirchhoff matrix and can call calculation of normal modes
    '''

    # Read .pdb file and filter CA atoms
    mmtf_file = mmtf.MMTFFile.read(rcsb.fetch(id, "mmtf"))
    whole_structure = mmtf.get_structure(mmtf_file, model=1, include_bonds=True)
    protein = whole_structure[struc.filter_amino_acids(whole_structure)]
    ca = protein[(protein.atom_name == "CA") & (protein.element == "C")]
    
    # Define force field and GNM object
    ff = springcraft.InvariantForceField(cutoff_distance=cutoff)
    anm = springcraft.ANM(ca, ff)

    # Return model with all pre-processing
    return anm    

def pseudo_fluctuation_measure(eval, evec, K = 10, funct = (lambda x: 1/x)):
    '''
    Function that calculates pseudo fluctuation measure calculated as sum of eigenvectors scaled by softmaxed eigenvalues.

    Arguments:
        eval : numpy.array(float) - eigenvalues of the structure
        evec : numpy.array(numpy.array(float)) - eigenvectors of the structure
        K : int - number of modes to be considered
        funct : lambda - function directing which property to pass to softmax 
    Returns:
        pseudo_fluc : torch.Tensor - pseudo fluctuation value from eigenvalues and eigenvectors 
    '''
    eval = eval[1:K]
    w = scipy.special.softmax(funct(eval))
    evec = evec[1:K]
    evec = evec**2
    evec = evec.transpose()
    pseudo_fluc = (w*evec).sum(axis=1)
    return torch.from_numpy(pseudo_fluc)

def plot_flexibility_value(id, cutoff, k = 10, flex: Literal["msqf", "bfact", "pseudo"] = "msqf"):
    '''
    Function that plots flexibility value against all residues.

    Arguments:
        id : string - PDB ID of the structure
        cutoff : double - cutoff distance for NMA analysis
        k : int - number of modes to be considered
        flex : Literal - type of flexibility value to plot  
    '''
    gnm = calculate_normal_modes_gnm_from_id(id, cutoff)
    if flex == "msqf":
        msqf = gnm.mean_square_fluctuation(mode_subset = np.array([i for i in range(1, k)]))
        fig = plt.figure(figsize=(8.0, 4.0), constrained_layout=True)
        grid = fig.add_gridspec(nrows=1, ncols=2)
        ax = fig.add_subplot(grid[0, :])

        biotite_c = biotite.colors["orange"]

        ax.bar(x=np.arange(1, len(msqf)+1), height=msqf, color=biotite_c)
        ax.set_xlabel("Amino Acid Residue ID", size=16)
        ax.set_ylabel("Mean squared fluctuation / A.U.", size=16)

        plt.show()
    elif flex == "bfact":
        bfact = gnm.bfactor(mode_subset = np.array([i for i in range(1, k)]))
        fig = plt.figure(figsize=(8.0, 4.0), constrained_layout=True)
        grid = fig.add_gridspec(nrows=1, ncols=2)
        ax = fig.add_subplot(grid[0, :])

        biotite_c = biotite.colors["orange"]

        ax.bar(x=np.arange(1, len(bfact)+1), height=bfact, color=biotite_c)
        ax.set_xlabel("Amino Acid Residue ID", size=16)
        ax.set_ylabel("B Factor", size=16)
    elif flex == "pseudo":
        eigenval, eigenvec = gnm.eigen() 
        pseudo_fluc = pseudo_fluctuation_measure(eigenval, eigenvec, k, lambda x: 1/x)
        fig = plt.figure(figsize=(8.0, 4.0), constrained_layout=True)
        grid = fig.add_gridspec(nrows=1, ncols=2)
        ax = fig.add_subplot(grid[0, :])

        biotite_c = biotite.colors["orange"]

        ax.bar(x=np.arange(1, len(pseudo_fluc)+1), height=pseudo_fluc, color=biotite_c)
        ax.set_xlabel("Amino Acid Residue ID", size=16)
        ax.set_ylabel("Pseudo Fluctuation Measure", size=16)
    else:
        raise ValueError("Not appropriate flex type.")
    
def plot_eigval_freq(id, cutoff):
    '''
    Function that plots eigenvalues and frequencies agains all normal modes.

    Arguments:
        id : string - PDB ID of the structure
        cutoff : double - cutoff distance for NMA analysis
    '''
    gnm = calculate_normal_modes_gnm_from_id(id, cutoff)
    eigenval, eigenvec = gnm.eigen() 

    freq = gnm.frequencies()[1:]

    eigenval = eigenval[1:]
    fig = plt.figure(figsize=(8.0, 4.0), constrained_layout=True)
    grid = fig.add_gridspec(nrows=1, ncols=2)

    ax00 = fig.add_subplot(grid[0, 0])
    ax01 = fig.add_subplot(grid[0, 1])

    biotite_c = biotite.colors["orange"]

    ax00.bar(x=np.arange(1, len(eigenval)+1), height=eigenval, color=biotite_c)
    ax01.bar(x=np.arange(1, len(freq)+1), height=freq, color=biotite_c)

    ax00.set_xlabel("Mode", size=16)
    ax00.set_ylabel(r"Eigenvalue $\lambda$", size=16)
    ax01.set_xlabel("Mode", size=16)
    ax01.set_ylabel(r"Frequency $\nu$ / A.U.", size=16)

    plt.show()