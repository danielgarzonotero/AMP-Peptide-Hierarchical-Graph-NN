
import torch
import torch.nn.functional as F
import numpy as np




def train(model, device, dataloader, optim, epoch):
    model.train()
    
    loss_func = torch.nn.BCEWithLogitsLoss() #TODO
    loss_collect = 0

    # Looping over the dataloader allows us to pull out input/output data:
    for batch in dataloader:
        # Zero out the optimizer:        
        optim.zero_grad()
        batch = batch.to(device)

        # Make a prediction:
        pred = model(batch)
        
        # Convertir la lista de etiquetas a un tensor
        labels_list = [label_representation(y, device) for y in batch.y]
        y_labels_tensor = torch.stack(labels_list)


        # Calculate the loss:
        loss = loss_func(pred.double(), y_labels_tensor.double())

        # Backpropagation:
        loss.backward()
        optim.step()

        # Calculate the loss and add it to our total loss
        loss_collect += loss.item()  # loss summed across the batch

    # Return our normalized losses so we can analyze them later:
    loss_collect /= len(dataloader.dataset)
    
    print(
        "Epoch:{}   Training dataset:   Loss per Datapoint: {:.3f}%".format(
            epoch, loss_collect * 100
        )
    ) 
    return loss_collect    

def validation(model, device, dataloader, epoch):

    model.eval()
    loss_collect = 0
    loss_func = torch.nn.BCEWithLogitsLoss()
        
    # Remove gradients:
    with torch.no_grad():

        for batch in dataloader:
            
            batch = batch.to(device)
    
            # Make a prediction:
            pred = model(batch)
            
            # Convertir la lista de etiquetas a un tensor
            labels_list = [label_representation(y, device) for y in batch.y]
            y_labels_tensor = torch.stack(labels_list)


            # Calculate the loss:
            loss = loss_func(pred.double(), y_labels_tensor.double())  

            # Calculate the loss and add it to our total loss
            loss_collect += loss.item()  # loss summed across the batch

    loss_collect /= len(dataloader.dataset)
    
    # Print out our test loss so we know how things are going
    print(
        "Epoch:{}   Validation dataset: Loss per Datapoint: {:.3f}%".format(
            epoch, loss_collect * 100
        )
    )  
    print('---------------------------------------')     
    # Return our normalized losses so we can analyze them later:
    return loss_collect


def predict_test(model, dataloader, device, weights_file):
    
    model.eval()
    model.load_state_dict(torch.load(weights_file))

    X_all = []
    y_all = []
    pred_all = []

    # Remove gradients:
    with torch.no_grad():

        # Looping over the dataloader allows us to pull out input/output data:
        for batch in dataloader:

            # Make a prediction:
            pred = model(batch.to(device))
            
            labels_list = [label_representation(y, device) for y in batch.y]
            y_labels_tensor = torch.stack(labels_list)
            
            X_all.append(batch.x.to(device))
            y_all.append(y_labels_tensor)
            pred_all.append(pred)

    # Concatenate the lists of tensors into a single tensor
    X_all = torch.cat(X_all, dim=0)
    y_all = torch.cat(y_all, dim=0)
    pred_all = torch.cat(pred_all, dim=0)

    return X_all, y_all, pred_all


def label_representation(y, device):
    if y == 1:
        
        return torch.tensor(np.array([1]), dtype=torch.long, device=device)
    elif y == 0:
        return torch.tensor(np.array([0]), dtype=torch.long, device=device)
    else:
        raise ValueError("Invalid value for y. It should be either 0 or 1.")
    