
import torch
import torch.nn.functional as F
import numpy as np


def train(model, device, dataloader, optim, epoch, type_dataset):
    model.train()
    
    loss_func = torch.nn.BCEWithLogitsLoss() 
    loss_collect = 0
    
    if  type_dataset== 'training':
        aminoacids_features_dict = importar_dicionario(type_dataset, 'aminoacids_features')
        blosum62_dict = importar_dicionario(type_dataset, 'blosum62')
    elif type_dataset == 'validation':
        aminoacids_features_dict = importar_dicionario(type_dataset, 'aminoacids_features')
        blosum62_dict = importar_dicionario(type_dataset, 'blosum62')
    elif type_dataset == 'testing':
        aminoacids_features_dict = importar_dicionario(type_dataset, 'aminoacids_features')
        blosum62_dict = importar_dicionario(type_dataset, 'blosum62')
    else:
        raise ValueError(f"Unsupported dataset: {type_dataset}")
    
    # Looping over the dataloader allows us to pull out input/output data:
    for batch in dataloader:
        # Zero out the optimizer:        
        optim.zero_grad()
        batch = batch.to(device)
        x, edge_index,  edge_attr, idx_batch, cc, monomer_labels = batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.cc, batch.monomer_labels
        
        # Make a prediction:
        pred = model(
                    x,
                    edge_index,
                    edge_attr,
                    aminoacids_features_dict,
                    blosum62_dict,
                    idx_batch,
                    cc,
                    monomer_labels
        )
        
        
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
        "Epoch:{}   Training dataset:   Loss per Datapoint: {:.7f}%".format(
            epoch, loss_collect * 100
        )
    ) 
    return loss_collect    

def validation(model, device, dataloader, epoch, type_dataset):

    model.eval()
    loss_collect = 0
    loss_func = torch.nn.BCEWithLogitsLoss() 
    
    if  type_dataset== 'training':
        aminoacids_features_dict = importar_dicionario(type_dataset, 'aminoacids_features')
        blosum62_dict = importar_dicionario(type_dataset, 'blosum62')
    elif type_dataset == 'validation':
        aminoacids_features_dict = importar_dicionario(type_dataset, 'aminoacids_features')
        blosum62_dict = importar_dicionario(type_dataset, 'blosum62')
    elif type_dataset == 'testing':
        aminoacids_features_dict = importar_dicionario(type_dataset, 'aminoacids_features')
        blosum62_dict = importar_dicionario(type_dataset, 'blosum62')
    else:
        raise ValueError(f"Unsupported dataset: {type_dataset}")
    
    # Remove gradients:
    with torch.no_grad():

        for batch in dataloader:
            
            batch = batch.to(device)
            x, edge_index,  edge_attr, idx_batch, cc, monomer_labels = batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.cc, batch.monomer_labels
            
            # Make a prediction:
            pred = model(
                        x,
                        edge_index,
                        edge_attr,
                        aminoacids_features_dict,
                        blosum62_dict,
                        idx_batch,
                        cc,
                        monomer_labels
            )
            
            
            # Calculate the loss:
            loss = loss_func(pred.double(), batch.y.double())  

            # Calculate the loss and add it to our total loss
            loss_collect += loss.item()  # loss summed across the batch

    loss_collect /= len(dataloader.dataset)
    
    # Print out our test loss so we know how things are going
    print(
        "Epoch:{}   Validation dataset: Loss per Datapoint: {:.4f}%".format(
            epoch, loss_collect * 100
        )
    )  
    print('---------------------------------------')     
    # Return our normalized losses so we can analyze them later:
    return loss_collect


def predict_test(model, dataloader, device, weights_file, threshold, type_dataset):
    
    
    if  type_dataset== 'training':
        aminoacids_features_dict = importar_dicionario(type_dataset, 'aminoacids_features')
        blosum62_dict = importar_dicionario(type_dataset, 'blosum62')
        id_secuencias = importar_dicionario(type_dataset, 'sequences')
    elif type_dataset == 'validation':
        aminoacids_features_dict = importar_dicionario(type_dataset, 'aminoacids_features')
        blosum62_dict = importar_dicionario(type_dataset, 'blosum62')
        id_secuencias = importar_dicionario(type_dataset, 'sequences')
    elif type_dataset == 'testing':
        aminoacids_features_dict = importar_dicionario(type_dataset, 'aminoacids_features')
        blosum62_dict = importar_dicionario(type_dataset, 'blosum62')
        id_secuencias = importar_dicionario(type_dataset, 'sequences')
    else:
        raise ValueError(f"Unsupported dataset: {type_dataset}")
    
    model.eval()
    model.load_state_dict(torch.load(weights_file))
    
    x_all = []
    y_all = []
    pred_all = []
    
    pred_all_csv = []
    
    # Remove gradients:
    with torch.no_grad():

        # Looping over the dataloader allows us to pull out input/output data:
        for batch in dataloader:
            
            batch = batch.to(device)
            x, edge_index,  edge_attr, idx_batch, cc, monomer_labels = batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.cc, batch.monomer_labels
            
            # Make a prediction:
            pred = model(
                        x,
                        edge_index,
                        edge_attr,
                        aminoacids_features_dict,
                        blosum62_dict,
                        idx_batch,
                        cc,
                        monomer_labels
            )
            
            pred_sigmoid = torch.sigmoid(pred) #to be able to round and saving in a csv file as prediction results
            
            x_all.extend([id_secuencias[cci.item()] for cci in batch.cc])
            y_all.append(batch.y.double())
            pred_all.append(pred)
            
            pred_all_csv.append(pred_sigmoid)
            
            

    # Concatenate the lists of tensors into a single tensor
    y_all = torch.cat(y_all, dim=0)
    pred_all = torch.cat(pred_all, dim=0)
    
    #This is to export the prediction rounded based on the threshold
    pred_all_csv = torch.cat(pred_all_csv, dim=0)
    pred_all_csv = [custom_round(pred, threshold) for pred in pred_all_csv]
    
    return x_all, y_all, pred_all, pred_all_csv


def custom_round(pred, threshold):
    return 1 if pred >= threshold else 0


def importar_dicionario(type_dataset, type_dictionario):
    return torch.load('data/dictionaries/{}/{}_dict.pt'.format(type_dataset, type_dictionario))


