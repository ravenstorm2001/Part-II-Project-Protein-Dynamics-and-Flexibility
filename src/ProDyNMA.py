# This file is just using ProDy to extract normal modes, but switched to springcraft for simplicity

from prody import GNM
from prody import ANM
from prody import parsePDB 

def calculate_modes_gnm_cpu(protein_name, cutoff = 7.0, num_modes=20, gamma = 1):
    """
    Function to calculate normal modes on CPU

    Arguments:  
        protein_name : string - PDB id of a protein
        cutoff : double - cutoff distance for GNM algorithm
        num_modes : integer - number of lowest non-trivial modes calculated
        gamma : double - constant for potential energy calculation
    Outputs:
        (eigenvals, eigenvecs) : (array(double), array(array(double))) - calculated modes and eigenvalues 
    """
    protein = parsePDB(protein_name)
    calphas = protein.select('calpha')

    gnm = GNM()
    gnm.buildKirchhoff(coords=calphas, cutoff=cutoff, gamma=gamma)

    gnm.calcModes(num_modes, zeros=False)
    return (gnm.getEigvals(), gnm.getEigvecs())

def calculate_modes_anm_cpu(protein_name, cutoff = 7.0, num_modes=20, gamma = 1):
    """
    Function to calculate normal modes on CPU

    Arguments:  
        protein_name : string - PDB id of a protein
        cutoff : double - cutoff distance for GNM algorithm
        num_modes : integer - number of lowest non-trivial modes calculated
        gamma : double - constant for potential energy calculation
    Outputs:
        (eigenvals, eigenvecs) : (array(double), array(array(double))) - calculated modes and eigenvalues 
    """
    protein = parsePDB(protein_name)
    calphas = protein.select('calpha')

    anm = ANM()
    anm.buildHessian(coords=calphas, cutoff=cutoff, gamma=gamma)

    anm.calcModes(num_modes, zeros=False)
    return (anm.getEigvals(), anm.getEigvecs())