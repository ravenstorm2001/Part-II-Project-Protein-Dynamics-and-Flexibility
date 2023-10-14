# File for making experiment functions.
import time
import torch
import torch.nn.functional as F
import numpy as np

from torcheval.metrics.functional import multiclass_accuracy
from tqdm import tqdm

def train(model, train_loader, optimizer, device, num_classes):
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
        loss = F.cross_entropy(y_pred, torch.reshape(data.y, (-1 ,num_classes)))
        loss.backward()
        loss_all += loss.item()
        optimizer.step()
    return loss_all / len(train_loader.dataset)

def eval_accuracy(model, loader, device, num_classes):
    """
    Function for evaluating the model.

    Arguments:
        model: torch.nn.Module - model we want to evaluate
        loader: torch_geometric.loader.DataLoader - data wrapped up in a PyG loader
        device: str - device on which to perform evaluation
        num_classes: int - number of classes in output space
    Return:
        error: float - error defined by specific metric  
    """
    model.eval()
    error = 0

    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            y_pred = model(data)  # [batch_size * num_classes]
            # Accuracy on a batch
            error += multiclass_accuracy(y_pred, torch.argmax(torch.reshape(data.y, (-1 ,num_classes)), dim = 1))
    return error / len(loader)

def eval_metrics(model, dataset, device:str = "gpu", num_classes:int = 384):
    """
    Function for evaluating the model.

    Arguments:
        model: torch.nn.Module - model we want to evaluate
        loader: torch_geometric.loader.DataLoader - data wrapped up in a PyG loader
        device: str - device on which to perform evaluation
        num_classes: int - number of classes in output space
    Return:
        error: float - error defined by specific metric  
    """
    model.eval()
    tp = [0 for i in range(num_classes)]
    tn = [0 for i in range(num_classes)]
    fp = [0 for i in range(num_classes)]
    fn = [0 for i in range(num_classes)]

    for data in tqdm(dataset):
        with torch.no_grad():
            y_pred = model(data)  # [batch_size * num_classes]
            pred = torch.argmax(y_pred, dim=1)
            actual = torch.argmax(torch.reshape(data.y, (-1 ,num_classes)), dim = 1)
            if pred == actual:
                tp[pred] += 1
                for i in range(num_classes):
                    if i!=pred:
                        tn[i]+=1
            else:
                fp[pred] += 1
                fn[actual] += 1
                for i in range(num_classes):
                    if i!=pred and i!=actual:
                        tn[i]+=1
    return tp, tn, fp, fn

def eval_corr(model, dataset, device:str = "gpu", num_classes:int = 6):
    """
    Function for evaluating the model.

    Arguments:
        model: torch.nn.Module - model we want to evaluate
        loader: torch_geometric.loader.DataLoader - data wrapped up in a PyG loader
        device: str - device on which to perform evaluation
        num_classes: int - number of classes in output space
    Return:
        error: float - error defined by specific metric  
    """
    model.eval()
    corr_matrix = [[0 for i in range(num_classes)] for j in range(num_classes)]

    for data in tqdm(dataset):
        with torch.no_grad():
            y_pred = model(data)  # [batch_size * num_classes]
            pred = torch.argmax(y_pred, dim=1)
            actual = torch.argmax(torch.reshape(data.y, (-1 ,num_classes)), dim = 1)
            corr_matrix[actual][pred]+=1
    return corr_matrix


def run_experiment(model, model_name, train_loader, val_loader, test_loader, n_epochs=100):
    """
    Function for running the experiment.

    Arguments:
        model: torch.nn.Module - model we want to run the experiment on
        train_loader: torch_geometric.loader.DataLoader - train data wrapped up in a PyG loader
        val_loader: torch_geometric.loader.DataLoader - validation data wrapped up in a PyG loader
        test_loader: torch_geometric.loader.DataLoader - test data wrapped up in a PyG loader
        n_epochs: int - num of epochs we want our training to be done.
    Return:
        loss: float - loss on training  
    """
    print(f"Running experiment for {model_name}, training on {len(train_loader.dataset)} samples for {n_epochs} epochs.")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\nDevice:")
    if(torch.cuda.is_available()):
        print("GPU")
    else:
        print("CPU")
    print("\nModel architecture:")
    print(model)
    
    total_param = 0
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
    print(f'Total parameters: {total_param}')
    
    model = model.to(device)

    # Adam optimizer with LR 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    print("\nStart training:")
    best_accuracy = None
    perf_per_epoch = [] # Track Test/Val MAE vs. epoch (for plotting)
    t = time.time()
    for epoch in range(1, n_epochs+1):
        # Train model for one epoch, return avg. training loss
        loss = train(model, train_loader, optimizer, device, 384)
        
        # Evaluate model on validation set
        val_accuracy = eval_accuracy(model, val_loader, device, 384)
        
        if best_accuracy is None or val_accuracy >= best_accuracy:
            # Evaluate model on test set if validation metric improves
            test_accuracy = eval_accuracy(model, test_loader, device, 384)
            best_accuracy = val_accuracy

        if epoch % 10 == 0:
            # Print and track stats every 10 epochs
            print(f'Epoch: {epoch:03d}, LR: {0.001:5f}, Loss: {loss:.7f}, '
                  f'Val Accuracy: {val_accuracy:.7f}, Test Accuracy: {test_accuracy:.7f}')
        
        perf_per_epoch.append((test_accuracy, val_accuracy, epoch, model_name))
    
    t = time.time() - t
    train_time = t/60
    print(f"\nDone! Training took {train_time:.2f} mins. Best validation accuracy: {best_accuracy:.7f}, corresponding test accuracy: {test_accuracy:.7f}.")
    
    return best_accuracy, test_accuracy, train_time, perf_per_epoch