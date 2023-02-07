# This file is used to calculate/extract and plot normal modes from a file or from PDBID

import biotite.structure as struc
import biotite.structure.io.mmtf as mmtf
import biotite.database.rcsb as rcsb
import biotite.structure.io as strucio
import springcraft

def calculate_normal_modes_gnm_from_file(path, cutoff):
    '''
    Function that calculates eigenvalues and eigenvectors of Kirchhoff matrix using GNM method from .pdb file.

    Arguments:
        path : string - Path to .pdb file containing structure we want to explore
        cutoff : double - Angstrom distance under which two CA atoms are represented as connected
    Returns:
        eigenvalues : array(double) - Eigenvalues of Kirchhoff matrix
        eigenvectors : matrix(double) - Eigenvectors of Kirchhoff matrix represented as rows of the matrix returned   
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
        eigenvalues : array(double) - Eigenvalues of Kirchhoff matrix
        eigenvectors : matrix(double) - Eigenvectors of Kirchhoff matrix represented as rows of the matrix returned   
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
        eigenvalues : array(double) - Eigenvalues of Kirchhoff matrix
        eigenvectors : matrix(double) - Eigenvectors of Kirchhoff matrix represented as rows of the matrix returned   
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
        eigenvalues : array(double) - Eigenvalues of Kirchhoff matrix
        eigenvectors : matrix(double) - Eigenvectors of Kirchhoff matrix represented as rows of the matrix returned   
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