# File for Layer and Model implementations
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import pandas as pd
import os

from typing import Callable, Optional, Literal
from torch.nn import Linear, ReLU, BatchNorm1d, Module, Sequential, SiLU
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_scatter import scatter
from torch_geometric.loader import DataLoader
from src.data_loaders import ProteinDataset, load_labels
from src.data_preprocessing import calculate_all_structures_and_store
from libraries.lmdb_dataset import LMDBDataset
from torchmetrics import F1Score, Accuracy
from torch.utils.data import ConcatDataset

class EGNNLayer(MessagePassing):
    def __init__(self, emb_dim=64, edge_dim=32, aggr='add'):
        """ 
        EGNN Message Passing Neural Network Layer

        This layer is equivariant in 3D rotations and translations and permutations with respect to the input. 

        Args:
            emb_dim: (int) - hidden dimension `d`
            edge_dim: (int) - edge feature dimension `d_e`
            aggr: (str) - aggregation function `\oplus` (sum/mean/max)
        """
        super().__init__(aggr=aggr)

        self.emb_dim = emb_dim
        self.edge_dim = edge_dim

        self.mlp_msg = Sequential(
            Linear(2*emb_dim + edge_dim + 1, emb_dim), BatchNorm1d(emb_dim), SiLU(),
            Linear(emb_dim, emb_dim), BatchNorm1d(emb_dim), SiLU()
          )  
        
        self.mlp_upd_features = Sequential(
            Linear(2*emb_dim, emb_dim), BatchNorm1d(emb_dim), SiLU(), 
            Linear(emb_dim, emb_dim)
          )
        
        self.mlp_upd_coord = Sequential(
            Linear(2*emb_dim + edge_dim + 1, emb_dim), BatchNorm1d(emb_dim), SiLU(), 
            Linear(emb_dim, 1)
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
    
    def aggregate(self, msgs, index):
        """The `aggregate` function aggregates the messages from neighboring nodes,
        according to the chosen aggregation function ('sum' by default).

        Args:
            inputs: (e, d) - messages `m_ij` from destination to source nodes
            index: (e, 1) - list of source nodes for each edge/message in `input`

        Returns:
            aggr_out: (n, d) - aggregated messages `m_i`
        """
        msgs_x, msgs_coord = msgs
        return (scatter(msgs_x, index, dim=self.node_dim, reduce=self.aggr),\
                scatter(msgs_coord, index, dim=self.node_dim, reduce=self.aggr)/(msgs_coord.size()[0]-1))
    
    def update(self, aggr_out, h):
      """The `update()` function computes the final node features by combining the 
        aggregated messages with the initial node features.

        Args:
            aggr_out: (n, d) - aggregated messages `m_i`
            h: (n, d) - initial node features

        Returns:
            upd_out: (n, d) - updated node features passed through MLP `\phi`
      """
      upd_out_features = torch.cat([h, aggr_out[0]], dim=-1)
      return (self.mlp_upd_features(upd_out_features),aggr_out[1])

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(emb_dim={self.emb_dim}, aggr={self.aggr})')

class PlEGNNModel(pl.LightningModule):
    def __init__(self,
                num_layers=4, 
                emb_dim=64, 
                in_dim=31, 
                edge_dim=32, 
                out_dim=6, 
                num_classes=6, 
                lr: float = 0.0001,
                rate_decay: float = 0.98):
        """Message Passing Neural Network model for graph classification

        This model uses both node features and coordinates as inputs, and
        is invariant to 3D rotations and translations.

        Args:
            num_layers: (int) - number of message passing layers `L`
            emb_dim: (int) - hidden dimension `d`
            in_dim: (int) - initial node feature dimension `d_n`
            edge_dim: (int) - edge feature dimension `d_e`
            out_dim: (int) - output dimension (fixed to 1)
            num_classes: (int) - number of classes in data
        """
        super().__init__()
        
        # Linear projection for initial node features
        # dim: in_dim -> emb_dim
        self.lin_in = Linear(in_dim, emb_dim) 
        
        # Stack of equivariant MPNN layers
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(EGNNLayer(emb_dim, edge_dim, aggr='add'))

        # Global pooling/readout function `R` (mean pooling)
        # PyG handles the underlying logic via `global_mean_pool()`
        self.pool = global_mean_pool

        # Linear prediction head
        # dim: d -> out_dim
        self.lin_pred = Linear(emb_dim, out_dim)

        self.num_classes = num_classes

        self.train_f1 = F1Score(task="multiclass", num_classes=self.num_classes, average = "macro")
        self.train_acc = Accuracy(task="multiclass", num_classes=self.num_classes, average = "micro")
        self.val_f1 = F1Score(task="multiclass", num_classes=self.num_classes, average = "macro")
        self.val_acc = Accuracy(task="multiclass", num_classes=self.num_classes, average = "micro")
        self.test_f1 = F1Score(task="multiclass", num_classes=self.num_classes, average = "macro")
        self.test_acc = Accuracy(task="multiclass", num_classes=self.num_classes, average = "micro")
        self.lr = lr
        self.rate_decay = rate_decay

    def forward(self, data):
        """
        Args:
            data: (PyG.Data) - batch of PyG graphs

        Returns: 
            out: (batch_size, out_dim) - prediction for each graph
        """
        h = self.lin_in(torch.cat((data.node_s.to(torch.float32),data.node_type.to(torch.float32)), dim=1)) # (n, d_n) -> (n, d)
        pos = data.x

        for conv in self.convs:
            h_upd, pos_upd = conv(h, pos, data.edge_index, data.edge_s) # (n, d) -> (n, d)
            
            h = h + h_upd 
            pos = pos + pos_upd

        h_graph = self.pool(h, data.batch) # (n, d) -> (batch_size, d)

        logits = self.lin_pred(h_graph) # (batch_size, d) -> (batch_size, num_classes)

        return logits
    
    def training_step(self, batch, batch_idx):
        y_hat = self(batch)
        loss = F.cross_entropy(y_hat, batch.y.view(-1, self.num_classes))
        pred = y_hat.argmax(dim = 1)
        target = batch.y.view(-1, self.num_classes).argmax(dim = 1)
        self.train_acc(pred, target)
        self.train_f1(pred, target)
        self.log("train_macro_f1", self.train_f1, on_step=False, on_epoch=True, batch_size=batch.size()[0])
        self.log("train_micro_acc", self.train_acc, on_step=False, on_epoch=True, batch_size=batch.size()[0])
        self.log("train_loss", loss, on_step=False, on_epoch=True, batch_size=batch.size()[0])
        return loss
    
    def validation_step(self, batch, batch_idx):
        y_hat = self(batch)
        val_loss = F.cross_entropy(y_hat, batch.y.view(-1, self.num_classes))
        pred = y_hat.argmax(dim = 1)
        target = batch.y.view(-1, self.num_classes).argmax(dim = 1)
        self.val_acc(pred, target)
        self.val_f1(pred, target)
        self.log("val_macro_f1", self.val_f1, on_step=False, on_epoch=True, batch_size=batch.size()[0])
        self.log("val_micro_acc", self.val_acc, on_step=False, on_epoch=True, batch_size=batch.size()[0])
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, batch_size=batch.size()[0])

    def test_step(self, batch, batch_idx):
        y_hat = self(batch)
        test_loss = F.cross_entropy(y_hat, batch.y.view(-1, self.num_classes))
        pred = y_hat.argmax(dim = 1)
        target = batch.y.view(-1, self.num_classes).argmax(dim = 1)
        self.test_acc(pred, target)
        self.test_f1(pred, target)
        self.log("test_macro_f1", self.test_f1, on_step=False, on_epoch=True, batch_size=batch.size()[0])
        self.log("test_micro_acc", self.test_acc, on_step=False, on_epoch=True, batch_size=batch.size()[0])
        self.log("test_loss", test_loss, on_step=False, on_epoch=True, batch_size=batch.size()[0])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, self.rate_decay, verbose = False)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}] 
    
class EGNNDataModule(pl.LightningDataModule):
    def __init__(self,
                 labels_dir: str = "./data/enzyme_data/",
                 database_file: str = "./data/pdb_data",
                 store_data: str = "./data/preprocessed_data",
                 batch_size: int = 64,
                 flex: Optional[Literal["msqf", "bfact", "pseudo"]] = None,
                 num_workers: int = 1,
                 num_modes: int = 10,
                 num_classes: int = 384,
                 num_folds: int = 3,
                 without_fold: int = 0):
        super().__init__()
        self.labels_dir = labels_dir
        self.store_data = store_data
        self.batch_size = batch_size
        self.labels = load_labels(labels_dir + "chain_functions.txt")
        self.database = LMDBDataset(database_file)
        if flex is None:
            self.flex = "no_flex"
        else:
            self.flex = flex
        self.num_workers = num_workers
        self.num_modes = num_modes
        self.num_classes = num_classes
        self.num_folds = num_folds
        self.without_fold = without_fold
    
    def prepare_data(self) -> None:
        if os.path.exists(self.store_data + "/" + self.flex):
            print("Warning: Check normal mode value, as preparation is not done.")
            return None
        os.mkdir(self.store_data)
        os.mkdir(self.store_data + "/" + self.flex)

        for i in range(0, self.num_classes):
            os.mkdir(self.store_data + "/" + self.flex + "/" + str(i))
        
        calculate_all_structures_and_store(self.labels, self.database, self.store_data, self.num_classes, self.flex , None, self.num_modes)
        return None

    def setup(self, stage: str):
        if stage == "test":
            self.test_dataset = ProteinDataset(self.labels,
                                            self.labels_dir + "splits/split"+str(self.num_folds)+".txt",
                                            self.flex,
                                            self.store_data)
        else:
            training = []
            for i in range(self.num_folds):
                if i!=self.without_fold:
                    training.append(ProteinDataset(self.labels,
                                                self.labels_dir + "splits/split" + str(i) + ".txt",
                                                self.flex,
                                                self.store_data))
            self.train_dataset = ConcatDataset(training)
            self.val_dataset = ProteinDataset(self.labels,
                                            self.labels_dir + "splits/split"+str(self.without_fold)+".txt",
                                            self.flex,
                                            self.store_data)
            self.test_dataset = ProteinDataset(self.labels,
                                            self.labels_dir + "splits/split"+str(self.num_folds)+".txt",
                                            self.flex,
                                            self.store_data)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          pin_memory = True,
                          num_workers = self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          pin_memory = True,
                          num_workers = self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          pin_memory = True,
                          num_workers = self.num_workers)
    