# File for making experiment functions.

import torch.nn.functional as F

def train(model, train_loader, optimizer, device):
    """
    Function for training the model.

    Arguments:
        model: torch.nn.Module - model we want to train
        train_loader: torch_geometric.loader.DataLoader - data wrapped up in a PyG loader
        optimizer: torch.optim object - optimizer for gradient descent step
        device: str - device on which to perform training
    Return:
        loss: float - loss on training  
    """
    model.train()
    loss_all = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        y_pred = model(data)
        loss = F.cross_entropy(y_pred, data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return loss_all / len(train_loader.dataset)