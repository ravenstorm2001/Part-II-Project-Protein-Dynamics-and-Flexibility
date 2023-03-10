# File to add unit tests for models and layers
import torch

from torch_geometric.utils import to_dense_adj, dense_to_sparse
from scipy.stats import ortho_group
from src.Models import EGNNSiLUModel, EGNNReLUModel

def permute_graph(data, perm):
    """Helper function for permuting PyG Data object attributes consistently.
    """
    # Permute the node attribute ordering
    data.x = data.x[perm]
    data.node_type = data.node_type[perm]
    data.node_s = data.node_s[perm]
    data.batch = data.batch[perm]

    # Permute the edge index
    adj = to_dense_adj(data.edge_index)
    adj = adj[:, perm, :]
    adj = adj[:, :, perm]
    data.edge_index = dense_to_sparse(adj)[0]

    # Note: 
    # (1) While we originally defined the permutation matrix P as only having 
    #     entries 0 and 1, its implementation via `perm` uses indexing into 
    #     torch tensors, instead. 
    # (2) It is cumbersome to permute the edge_attr, so we set it to constant 
    #     dummy values. For any experiments beyond unit testing, all GNN models 
    #     use the original edge_attr.

    return data

def permutation_invariance_unit_test(module, dataloader):
    """Unit test for checking whether a module (GNN model) is 
    permutation invariant.
    """
    it = iter(dataloader)
    data = next(it)

    # Set edge_attr to dummy values (for simplicity)
    data.edge_s = torch.zeros(data.edge_s.shape)

    # Forward pass on original example
    out_1 = module(data)

    # Create random permutation
    perm = torch.randperm(data.x.shape[0])
    data = permute_graph(data, perm)

    # Forward pass on permuted example
    out_2 = module(data)

    # Check whether output varies after applying transformations
    return torch.allclose(out_1, out_2, atol=1e-04)


def permutation_equivariance_unit_test(module, dataloader):
    """Unit test for checking whether a module (GNN layer) is 
    permutation equivariant.
    """
    it = iter(dataloader)
    data = next(it)

    # Set edge_attr to dummy values (for simplicity)
    data.edge_s= torch.zeros(data.edge_s.shape)

    # Forward pass on original example
    out_1 = module(torch.cat((data.node_s.to(torch.float32),data.node_type.to(torch.float32)), dim=1), data.x, data.edge_index, data.edge_s)

    # Create random permutation
    perm = torch.randperm(data.x.shape[0])
    data = permute_graph(data, perm)

    # Forward pass on permuted example
    out_2 = module(torch.cat((data.node_s.to(torch.float32),data.node_type.to(torch.float32)), dim=1), data.x, data.edge_index, data.edge_s)

    # Check whether output varies after applying transformations
    return torch.allclose(out_1[0][perm], out_2[0], atol=1e-04) and torch.allclose(out_1[1][perm], out_2[1], atol=1e-04)

def random_orthogonal_matrix(dim=3):
  """Helper function to build a random orthogonal matrix of shape (dim, dim)
  """
  Q = torch.tensor(ortho_group.rvs(dim=dim)).float()
  return Q


def rot_trans_invariance_unit_test(module, dataloader):
    """Unit test for checking whether a module (GNN model/layer) is 
    rotation and translation invariant.
    """
    it = iter(dataloader)
    data = next(it)

    out_1 = module(data)

    Q = random_orthogonal_matrix(dim=3)
    t = torch.rand(3)
    
    data.x = torch.mm(data.x, Q) + t.repeat(data.x.size(dim=0), 1)
    
    out_2 = module(data)

    return torch.allclose(out_1, out_2, atol=1e-04)

def rot_trans_equivariance_unit_test(module, dataloader):
    """Unit test for checking whether a module (GNN layer) is 
    rotation and translation equivariant.
    """
    it = iter(dataloader)
    data = next(it)

    _, pos_1 = module(torch.cat((data.node_s.to(torch.float32),data.node_type.to(torch.float32)), dim=1), data.x, data.edge_index, data.edge_s)

    Q = random_orthogonal_matrix(dim=3)
    t = torch.rand(3)
    # ============ YOUR CODE HERE ==============
    # Perform random rotation + translation on data.
    #
    data.x = torch.mm(data.x,Q) + t.repeat(data.x.size(dim=0), 1)
    # ==========================================

    # Forward pass on rotated + translated example
    _, pos_2 = module(torch.cat((data.node_s.to(torch.float32),data.node_type.to(torch.float32)), dim=1), data.x, data.edge_index, data.edge_s)

    return torch.allclose(torch.mm(pos_1,Q) + t.repeat(pos_1.size(dim=0), 1), pos_2, atol=1e-04)