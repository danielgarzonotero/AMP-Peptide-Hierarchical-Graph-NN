
import torch
import torch.nn.functional as F
import numpy as np




def train(model, device, dataloader, optim, epoch):
    model.train()
    
    loss_func = torch.nn.BCELoss() #TODO
    loss_collect = 0

    # Looping over the dataloader allows us to pull out input/output data:
    for batch in dataloader:
        # Zero out the optimizer:        
        optim.zero_grad()
        batch = batch.to(device)

        # Make a prediction:
        pred = model(batch)
        
        # Calculate the loss:
        loss = loss_func(pred.double(), batch.y.double())

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
    loss_func = torch.nn.BCELoss()
        
    # Remove gradients:
    with torch.no_grad():

        for batch in dataloader:
            
            batch = batch.to(device)
    
            # Make a prediction:
            pred = model(batch)
            

            # Calculate the loss:
            loss = loss_func(pred.double(), batch.y.double())  

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
    diccionario = torch.load('data/dictionaries/sequences_dict.pt')
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
            
            X_all.extend([diccionario[cci.item()] for cci in batch.cc])
            y_all.append(batch.y.double())
            pred_all.append(pred)

    # Concatenate the lists of tensors into a single tensor
    y_all = torch.cat(y_all, dim=0)
    pred_all = torch.cat(pred_all, dim=0)

    return X_all, y_all, pred_all


