
#%%
import time
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
import pandas as pd
from src.device import device_info
from src.data import GeoDataset
from src.model import GCN_Geo 
from src.process import train, validation, predict_test
from math import sqrt
from torchmetrics.classification import BinaryConfusionMatrix
from sklearn.metrics import roc_curve, auc


device_information = device_info()
print(device_information)
device = device_information.device

start_time = time.time()

## SET UP DATALOADERS: 

# Build starting dataset: 
dataset = GeoDataset(root='data')
print('Number of NODES features: ', dataset.num_features)
print('Number of EDGES features: ', dataset.num_edge_features)

finish_time_preprocessing = time.time()
time_preprocessing = (finish_time_preprocessing - start_time) / 60 #TODO


# Dataset Split Percent:
training_percentage = 0.70  #TODO
validation_percentage = 0.20  #TODO
test_percentage = 0.10  #TODO

n_train = int(len(dataset) * training_percentage)
n_val = int(len(dataset) * validation_percentage)
n_test = len(dataset) - n_train - n_val

# Define objetos de conjunto de entrenamiento, validación y prueba de PyTorch:
train_set, val_set, test_set = torch.utils.data.random_split(dataset,
                                                             [n_train, n_val, n_test],
                                                             generator=torch.Generator().manual_seed(42))

# Define dataloaders para conjuntos de entrenamiento, validación y prueba:
batch_size = 100  #TODO

dataloader = DataLoader(dataset, batch_size, shuffle=True)

train_dataloader = DataLoader(train_set, batch_size, shuffle=True)
val_dataloader = DataLoader(val_set, batch_size, shuffle=True)
test_dataloader = DataLoader(test_set, batch_size, shuffle=True)

## RUN TRAINING LOOP: 

# Train with a random seed to initialize weights:
torch.manual_seed(0)

# Set up model:
# Initial Inputs
initial_dim_gcn = dataset.num_features
edge_dim_feature = dataset.num_edge_features

hidden_dim_nn_1 = 15
hidden_dim_nn_2 = 10
hidden_dim_nn_3 = 0

hidden_dim_gat_0 = 50


hidden_dim_fcn_1 = 200
hidden_dim_fcn_2 = 200
hidden_dim_fcn_3 = 100 #TODO 


model = GCN_Geo(
                initial_dim_gcn,
                edge_dim_feature,
                hidden_dim_nn_1,
                hidden_dim_nn_2,
                hidden_dim_nn_3,

                hidden_dim_gat_0,
                
                hidden_dim_fcn_1,
                hidden_dim_fcn_2,
                hidden_dim_fcn_3,
            ).to(device)


# Set up optimizer:
learning_rate = 1E-3
weight_decay = 1E-5 #TODO
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

train_losses = []
val_losses = []

best_val_loss = float('inf')  # infinito

start_time_training = time.time()
number_of_epochs = 300

for epoch in range(1, number_of_epochs+1):
    train_loss = train(model, device, train_dataloader, optimizer, epoch)
    train_losses.append(train_loss)

    val_loss = validation(model, device, val_dataloader, epoch)
    val_losses.append(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        
        torch.save(model.state_dict(), "best_model_weights.pth")
     
finish_time_training = time.time()
time_training = (finish_time_training - start_time_training) / 60

#Lose curves
plt.plot(train_losses, label='Training loss', color='darkorange') 
plt.plot(val_losses, label='Validation loss', color='seagreen')  

# Aumentar el tamaño de la fuente en la leyenda
plt.legend(fontsize=14) 
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Traning and Validation Loss\nAMP Dataset', fontsize=17) #TODO
# Guardar la figura en formato PNG con dpi 216
plt.savefig('results/lose_curve.png', dpi=216)
plt.show()



#Testing:
weights_file = "best_model_weights.pth"
threshold=0.4

# ------------------------------------////////// Training set /////////////---------------------------------------------------
input_all_train, target_all_train, pred_prob_all_train, pred_all_csv_train = predict_test(model, train_dataloader, device, weights_file, threshold)

#Saving a CSV file with prediction values
prediction_train_set = {
                            'Sequence':input_all_train,
                            'Target': target_all_train.cpu().numpy(),
                            'Prediction':  pred_all_csv_train
                            }

df = pd.DataFrame(prediction_train_set)
df.to_excel('results/prediction_training_set.xlsx', index=False)

bcm = BinaryConfusionMatrix(task="binary", threshold=threshold, num_classes=2).to(device) #TODO
confusion_matrix = bcm(pred_prob_all_train, target_all_train)
confusion_matrix_np = confusion_matrix.detach().cpu().numpy()


true_negatives_train = confusion_matrix[0, 0].cpu().numpy()
false_positives_train = confusion_matrix[0, 1].cpu().numpy()
false_negatives_train = confusion_matrix[1, 0].cpu().numpy()
true_positives_train = confusion_matrix[1, 1].cpu().numpy()


# Añadir números a la matriz de confusión
cmap = plt.get_cmap('Blues')
plt.matshow(confusion_matrix_np, cmap=cmap)
plt.title('Confusion Matrix Plot - Training')
plt.colorbar()

# Añadir números a la matriz de confusión
for i in range(confusion_matrix_np.shape[0]):
    for j in range(confusion_matrix_np.shape[1]):
        plt.text(j, i, str(confusion_matrix_np[i, j]), ha='center', va='center', color='grey', fontsize=18)

plt.xlabel('Predicted Negative         Predicted Positive')
plt.ylabel('True Positive               True Negative')
plt.savefig('results/bcm_train.png', dpi=216)
plt.show()

# Cálculo de métricas de evaluación
accuracy_train = (true_positives_train + true_negatives_train) / (true_positives_train + true_negatives_train + false_positives_train + false_negatives_train)

precision_train = true_positives_train / (true_positives_train + false_positives_train)

recall_train = true_positives_train / (true_positives_train + false_negatives_train)

specificity_train = true_negatives_train / (true_negatives_train + false_positives_train)

f1_score_train = 2 * (precision_train * recall_train) / (precision_train + recall_train)

# Imprimir las métricas
print('///Evaluation Metrics - Trainning///\n') 
print(f"Accuracy: {accuracy_train:.3f}")
print(f"Precision: {precision_train:.3f}")
print(f"Recall: {recall_train:.3f}")
print(f"Specificity: {specificity_train:.3f}")
print(f"F1 Score: {f1_score_train:.3f}")

# Calcular la curva ROC
fpr, tpr, thresholds = roc_curve( target_all_train.cpu().numpy(), pred_prob_all_train.cpu().numpy())

# Calcular el área bajo la curva ROC (AUC)
roc_auc_train = auc(fpr, tpr)

# Trazar la curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc_train:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve - Trainning Set')
plt.legend(loc='lower right')
plt.show()

#-------------------------------------------- ////////// Validation Set //////////-------------------------------------------------
input_all_validation, target_all_validation, pred_prob_all_validation, pred_all_csv_validation = predict_test(model, val_dataloader, device, weights_file, threshold)

#Saving a CSV file with prediction values
prediction_validation_set = {
                            'Sequence':input_all_validation,
                            'Target': target_all_validation.cpu().numpy(),
                            'Prediction':  pred_all_csv_validation
                            }

df = pd.DataFrame(prediction_validation_set)
df.to_excel('results/prediction_validation_set.xlsx', index=False)


bcm = BinaryConfusionMatrix(task="binary", threshold=threshold, num_classes=2).to(device)  #TODO
confusion_matrix = bcm(pred_prob_all_validation, target_all_validation)
confusion_matrix_np = confusion_matrix.detach().cpu().numpy()

true_negatives_validation = confusion_matrix[0, 0].cpu().numpy() 
false_positives_validation = confusion_matrix[0, 1].cpu().numpy()
false_negatives_validation = confusion_matrix[1, 0].cpu().numpy() 
true_positives_validation = confusion_matrix[1, 1].cpu().numpy()

cmap = plt.get_cmap('Blues')
plt.matshow(confusion_matrix_np, cmap=cmap)
plt.title('Confusion Matrix Plot - Validation')
plt.colorbar()

# Añadir números a la matriz de confusión
for i in range(confusion_matrix_np.shape[0]):
    for j in range(confusion_matrix_np.shape[1]):
        plt.text(j, i, str(confusion_matrix_np[i, j]), ha='center', va='center', color='grey', fontsize=18)

plt.xlabel('Predicted Negative          Predicted Positive')
plt.ylabel('True Positive                True Negative')
plt.savefig('results/bcm_validation.png', dpi=216)
plt.show() 

# Cálculo de métricas de evaluación
accuracy_val = (true_positives_validation + true_negatives_validation) / (true_positives_validation + true_negatives_validation + false_positives_validation + false_negatives_validation)

precision_val = true_positives_validation / (true_positives_validation + false_positives_validation)

recall_val = true_positives_validation / (true_positives_validation + false_negatives_validation)

specificity_val = true_negatives_validation / (true_negatives_validation + false_positives_validation)

f1_score_val = 2 * (precision_val * recall_val) / (precision_val + recall_val)

# Imprimir las métricas
print('///Evaluation Metrics - Validation///\n') 
print(f"Accuracy: {accuracy_val:.3f}")
print(f"Precision: {precision_val:.3f}")
print(f"Recall: {recall_val:.3f}")
print(f"Specificity: {specificity_val:.3f}")
print(f"F1 Score: {f1_score_val:.3f}")

# Calcular la curva ROC
fpr, tpr, thresholds = roc_curve( target_all_validation.cpu().numpy(), pred_prob_all_validation.cpu().numpy())

# Calcular el área bajo la curva ROC (AUC)
roc_auc_validation = auc(fpr, tpr)

# Trazar la curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc_validation:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve - Validation Set')
plt.legend(loc='lower right')
plt.show()

# --------------------------------------------////////// Test Set //////////---------------------------------------------------
input_all_test, target_all_test, pred_prob_all_test, pred_all_csv_test = predict_test(model, test_dataloader, device, weights_file,threshold )

#Saving a CSV file with prediction values
prediction_test_set = {
                            'Sequence':input_all_test,
                            'Target': target_all_test.cpu().numpy(),
                            'Prediction':  pred_all_csv_test
                            }

df = pd.DataFrame(prediction_test_set)
df.to_excel('results/prediction_test_set.xlsx', index=False)

bcm = BinaryConfusionMatrix(task="binary",threshold=threshold, num_classes=2).to(device) #TODO
confusion_matrix = bcm(pred_prob_all_test, target_all_test)
confusion_matrix_np = confusion_matrix.detach().cpu().numpy()


true_negatives_test = confusion_matrix[0, 0].cpu().numpy()
false_positives_test = confusion_matrix[0, 1].cpu().numpy()
false_negatives_test= confusion_matrix[1, 0].cpu().numpy()
true_positives_test = confusion_matrix[1, 1].cpu().numpy()


# Añadir números a la matriz de confusión
cmap = plt.get_cmap('Blues')
plt.matshow(confusion_matrix_np, cmap=cmap)
plt.title('Confusion Matrix Plot - Test')
plt.colorbar()

# Añadir números a la matriz de confusión
for i in range(confusion_matrix_np.shape[0]):
    for j in range(confusion_matrix_np.shape[1]):
        plt.text(j, i, str(confusion_matrix_np[i, j]), ha='center', va='center', color='grey', fontsize=18)

plt.xlabel('Predicted Negative         Predicted Positive')
plt.ylabel('True Positive               True Negative')
plt.savefig('results/bcm_test.png', dpi=216)
plt.show()

# Cálculo de métricas de evaluación
accuracy_test = (true_positives_test + true_negatives_test) / (true_positives_test + true_negatives_test + false_positives_test+ false_negatives_test)

precision_test = true_positives_test / (true_positives_test + false_positives_test)

recall_test = true_positives_test / (true_positives_test + false_negatives_test)

specificity_test = true_negatives_test / (true_negatives_test + false_positives_test)

f1_score_test = 2 * (precision_test * recall_test) / (precision_test + recall_test)

# Imprimir las métricas
print('///Evaluation Metrics - Test///\n') 
print(f"Accuracy: {accuracy_test:.3f}")
print(f"Precision: {precision_test:.3f}")
print(f"Recall: {recall_test:.3f}")
print(f"Specificity: {specificity_test:.3f}")
print(f"F1 Score: {f1_score_test:.3f}")

# Calcular la curva ROC
fpr, tpr, thresholds = roc_curve( target_all_test.cpu().numpy(), pred_prob_all_test.cpu().numpy())

# Calcular el área bajo la curva ROC (AUC)
roc_auc_test= auc(fpr, tpr)

# Trazar la curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc_test:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve - Test Set')
plt.legend(loc='lower right')
plt.show()



#-----------------Times---------------------------
finish_time = time.time()
time_prediction = (finish_time - finish_time_training) / 60
total_time = (finish_time - start_time) / 60
print("\n //// Preprocessing time: {:3f} minutes ////".format(time_preprocessing))
print("\n //// Training time: {:3f} minutes ////".format(time_training))
print("\n //// Prediction time: {:3f} minutes ////".format(time_prediction))
print("\n //// Total time: {:3f} minutes ////".format(total_time))


#--------------------------------///////////Result DataFrame////////////---------------------------------------

data = {
    "Metric": [
        "number_features",
        "num_edge_features",
        "initial_dim_gcn ",
        "edge_dim_feature",
        "hidden_dim_nn_1 ",
        "hidden_dim_nn_2 ",
        "hidden_dim_nn_3 ",
        "hidden_dim_gat_0",
        "hidden_dim_fcn_1 ",
        "hidden_dim_fcn_2 ",
        "hidden_dim_fcn_3 ",
        "training_percentage %",
        "validation_percentage %",
        "test_percentage %",
        "batch_size", 
        "learning_rate",
        "weight_decay",
        "number_of_epochs",
        "threshold",
        "true_positives_train",
        "true_negatives_train",
        "false_positives_train",
        "false_negatives_train",
        "accuracy_train", 
        "precision_train", 
        "recall_train", 
        "specificity_train",
        "f1_score_train",
        "roc_auc_train", 
        "true_positives_validation",
        "true_negatives_validation",
        "false_positives_validation",
        "false_negatives_validation",
        "accuracy_val", 
        "precision_val", 
        "recall_val", 
        "specificity_val",
        "f1_score_val", 
        "roc_auc_validation",
        "true_positives_test",
        "true_negatives_test",
        "false_positives_test",
        "false_negatives_test",
        "accuracy_test", 
        "precision_test", 
        "recall_test", 
        "specificity_test",
        "f1_score_test", 
        "roc_auc_test",
        "time_preprocessing", 
        "time_training",
        "time_prediction",
        "total_time"
    ],
    "Value": [
        dataset.num_features,
        dataset.num_edge_features,
        initial_dim_gcn,
        edge_dim_feature ,
        hidden_dim_nn_1 ,
        hidden_dim_nn_2 ,
        hidden_dim_nn_3,
        hidden_dim_gat_0,
        hidden_dim_fcn_1 ,
        hidden_dim_fcn_2 ,
        hidden_dim_fcn_3 ,
        training_percentage*100,
        validation_percentage*100,
        test_percentage*100,
        batch_size,
        learning_rate,
        weight_decay,
        number_of_epochs,
        threshold,
        true_positives_train,
        true_negatives_train,
        false_positives_train,
        false_negatives_train,
        accuracy_train, 
        precision_train, 
        recall_train, 
        specificity_train,
        f1_score_train,
        roc_auc_train,
        true_positives_validation,
        true_negatives_validation,
        false_positives_validation,
        false_negatives_validation,
        accuracy_val, 
        precision_val, 
        recall_val, 
        specificity_val,
        f1_score_val,
        roc_auc_validation,
        true_positives_test,
        true_negatives_test,
        false_positives_test,
        false_negatives_test,
        accuracy_test, 
        precision_test, 
        recall_test, 
        specificity_test,
        f1_score_test,
        roc_auc_test,
        time_preprocessing, 
        time_training,
        time_prediction,
        total_time
    ],
    
}


df = pd.DataFrame(data)
df.to_csv('results/results.csv', index=False)



# %%
