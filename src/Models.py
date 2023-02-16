# File for Layer and Model implementations
import torch

from torch.nn import Linear, ReLU, BatchNorm1d, Module, Sequential
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_scatter import scatter

class EGNNMPNNLayer(MessagePassing):
    def __init__(self, emb_dim=64, edge_dim=32, aggr='add'):
        """EGNN Message Passing Neural Network Layer

        This layer is equivariant in 3D rotations and translations and permutations with respect to the input. 

        Args:
            emb_dim: (int) - hidden dimension `d`
            edge_dim: (int) - edge feature dimension `d_e`
            aggr: (str) - aggregation function `\oplus` (sum/mean/max)
        """
        super().__init__(aggr=aggr)

        self.emb_dim = emb_dim
        self.edge_dim = edge_dim

        self.mlp_msg = self.mlp_msg = Sequential(
            Linear(2*emb_dim + edge_dim + 1, emb_dim), BatchNorm1d(emb_dim), ReLU(),
            Linear(emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU()
          )  
        
        self.mlp_upd_features = Sequential(
            Linear(2*emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU(), 
            Linear(emb_dim, emb_dim), BatchNorm1d(emb_dim), ReLU()
          )
        self.mlp_upd_coord = Sequential(
            Linear(2*emb_dim + edge_dim + 1, emb_dim), BatchNorm1d(emb_dim), ReLU(), 
            Linear(emb_dim, 1), BatchNorm1d(1), ReLU()
          )

    def forward(self, h, pos, edge_index, edge_s):
        """
        The forward pass updates node features `h` via one round of message passing.

        Args:
            h: (n, d) - initial node features
            pos: (n, 3) - initial node coordinates
            edge_index: (e, 2) - pairs of edges (i, j)
            edge_s: (e, d_e) - edge features

        Returns:
            out: (n, d) - updated node features
        """
        out = self.propagate(edge_index, h=h, edge_s=edge_s, pos=pos)
        return out
     
    def message(self, h_i, h_j, pos_i, pos_j, edge_s):
      """The `message()` function constructs messages from source nodes j 
       to destination nodes i for each edge (i, j) in `edge_index`.
    
       Args:
           h_i: (e, d) - destination node features
           h_j: (e, d) - source node features
           pos_i: (e, 3) - destination node coordinates
           pos_j: (e, 3) - source node coordinates
           edge_s: (e, d_e) - edge features
        
       Returns:
           msg: (e, d) - messages `m_ij` passed through MLP `\psi`
      """ 
      msg = torch.concat([h_i, h_j, (pos_i-pos_j).norm(dim=1, p=2).reshape(-1,1), edge_s], dim=-1)
      return (self.mlp_msg(msg), self.mlp_upd_coord(msg)*(pos_i-pos_j))
    
    def aggregate(self, inputs, index):
        """The `aggregate` function aggregates the messages from neighboring nodes,
        according to the chosen aggregation function ('sum' by default).

        Args:
            inputs: (e, d) - messages `m_ij` from destination to source nodes
            index: (e, 1) - list of source nodes for each edge/message in `input`

        Returns:
            aggr_out: (n, d) - aggregated messages `m_i`
        """
        inputs_x, inputs_coord = inputs
        return (scatter(inputs_x, index, dim=self.node_dim, reduce=self.aggr),\
                scatter(inputs_coord, index, dim=self.node_dim, reduce=self.aggr))
    
    def update(self, aggr_out, h, pos):
      """The `update()` function computes the final node features by combining the 
        aggregated messages with the initial node features.

        Args:
            aggr_out: (n, d) - aggregated messages `m_i`
            h: (n, d) - initial node features

        Returns:
            upd_out: (n, d) - updated node features passed through MLP `\phi`
      """
      upd_out_features = torch.cat([h, aggr_out[0]], dim=-1)
      return (self.mlp_upd_features(upd_out_features),pos+aggr_out[1])

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(emb_dim={self.emb_dim}, aggr={self.aggr})')


class EGNNModel(Module):
    def __init__(self, num_layers=4, emb_dim=64, in_dim=8, edge_dim=32, out_dim=1):
        """Message Passing Neural Network model for graph classification

        This model uses both node features and coordinates as inputs, and
        is invariant to 3D rotations and translations.

        Args:
            num_layers: (int) - number of message passing layers `L`
            emb_dim: (int) - hidden dimension `d`
            in_dim: (int) - initial node feature dimension `d_n`
            edge_dim: (int) - edge feature dimension `d_e`
            out_dim: (int) - output dimension (fixed to 1)
        """
        super().__init__()
        
        # Linear projection for initial node features
        # dim: d_n -> d
        self.lin_in = Linear(in_dim, emb_dim)
        
        # Stack of equivariant MPNN layers
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(EGNNMPNNLayer(emb_dim, edge_dim, aggr='add'))
        
        # Global pooling/readout function `R` (mean pooling)
        # PyG handles the underlying logic via `global_mean_pool()`
        self.pool = global_mean_pool

        # Linear prediction head
        # dim: d -> out_dim
        self.lin_pred = Linear(emb_dim, out_dim)
        
    def forward(self, data):
        """
        Args:
            data: (PyG.Data) - batch of PyG graphs

        Returns: 
            out: (batch_size, out_dim) - prediction for each graph
        """
        h = self.lin_in(torch.cat((data.node_s.to(torch.float32),data.node_type.reshape(-1,1).to(torch.float32)), dim=1)) # (n, d_n) -> (n, d)
        pos = data.x

        for conv in self.convs:
            h, pos = conv(h, pos, data.edge_index, data.edge_s) # (n, d) -> (n, d)

        h_graph = self.pool(h, data.batch) # (n, d) -> (batch_size, d)

        logits = self.lin_pred(h_graph) # (batch_size, d) -> (batch_size, 384)

        return logits.view(-1)