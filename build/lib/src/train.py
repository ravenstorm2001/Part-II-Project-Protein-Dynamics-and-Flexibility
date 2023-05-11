import pytorch_lightning as pl
import torch

from src.pl_models import PlEGNNModel,EGNNDataModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from argparse import ArgumentParser


parser = ArgumentParser()

parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("-k", "--num_modes", type=int, default=10)
parser.add_argument("-lr","--learning_rate", type=float, default=0.0001)
parser.add_argument("-rd","--rate_decay", type=float, default=0.98)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--num_classes", type=int, default=384)

args = parser.parse_args()
print(args)

"""wandb_logger = WandbLogger(name='SiLU_pseudo_k_5_lr_1e-4_with_data', project='Testing')

early_stop_callback = EarlyStopping(monitor="val_macro_f1", min_delta=0.00, patience=25, verbose=False, mode="max")
checkpoint_callback = ModelCheckpoint(
    save_top_k=5,
    monitor="val_macro_f1",
    mode="max",
    dirpath="SiLUCheckpoints/",
    filename="pseudo_k_5-{epoch:02d}-{val_macro_f1:.2f}",
)

trainer = pl.Trainer(max_epochs=200, accelerator="gpu", logger = wandb_logger, callbacks=[checkpoint_callback, early_stop_callback])
model = PlEGNNModel(num_layers=4, emb_dim=64, in_dim=32, edge_dim=32, out_dim=384, num_classes=384,activation=torch.nn.SiLU)
datamodule = EGNNDataModule(flex = "pseudo", num_workers = 4)

trainer.fit(model, datamodule = datamodule)
"""