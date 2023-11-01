
import torch
import torch.nn as nn

def train(model, device, dataloader, optim, epoch):
    
    model.train()
    
    loss_func = nn.BCELoss() #TODO
    loss_collect = 0

    # Looping over the dataloader allows us to pull out or input/output data:
    # Enumerate allows us to also get the batch number:
    for batch in dataloader:

        # Zero out the optimizer:        
        optim.zero_grad()
        batch = batch.to(device)
        y_tensor_list = [torch.tensor([1, 0], dtype=torch.float32, device=device) if value == 1 
                       else torch.tensor([0, 1], dtype=torch.float32, device=device) for value in batch.y]
        
        y_label_tensor = torch.stack(y_tensor_list)
        
        # Make a prediction:
        pred = model(batch)

        # Calculate the loss:
        loss = loss_func(pred, y_label_tensor)  

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
    loss_func = nn.BCELoss()
        
    # Remove gradients:
    with torch.no_grad():

        
        for batch in dataloader:
            
            batch = batch.to(device)
            y_tensor_list = [torch.tensor([1, 0], dtype=torch.float32, device=device) if value == 1 
                       else torch.tensor([0, 1], dtype=torch.float32, device=device) for value in batch.y]
            
            y_label_tensor = torch.stack(y_tensor_list)
            
            # Make a prediction:
            pred = model(batch)

            # Calculate the loss:
            loss = loss_func(pred, y_label_tensor)  

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

            y_tensor_list = [torch.tensor([1, 0], dtype=torch.float32, device=device) if value == 1 
                       else torch.tensor([0, 1], dtype=torch.float32, device=device) for value in batch.y]
            y_label_tensor = torch.stack(y_tensor_list)
            
            X_all.append(batch.x.to(device))
            y_all.append(y_label_tensor)
            pred_all.append(pred)

    # Concatenate the lists of tensors into a single tensor
    X_all = torch.cat(X_all, dim=0)
    y_all = torch.cat(y_all, dim=0)
    pred_all = torch.cat(pred_all, dim=0)

    return X_all, y_all, pred_all


