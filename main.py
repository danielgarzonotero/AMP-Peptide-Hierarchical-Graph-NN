#%%
import time
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.explain import CaptumExplainer, Explainer

import pandas as pd
from src.device import device_info
from src.data import GeoDataset_1, GeoDataset_2,  GeoDataset_3
from src.model import GCN_Geo
from src.process import train, validation, predict_test
from math import sqrt
from torchmetrics.classification import BinaryConfusionMatrix
from sklearn.metrics import roc_curve, auc


''' print("PyTorch version:", torch.__version__)
print("PyTorch Geometric version:", torch_geometric.__version__) '''

device_information = device_info()
print(device_information)
device = device_information.device

start_time = time.time()

## SET UP DATALOADERS: 

# Build starting dataset: 
datasets = {
            'training_dataset': GeoDataset_1(root='data'),
            'validation_dataset': GeoDataset_2(root='data'),
            'testing_dataset': GeoDataset_3(root='data'),
            }


training_datataset = datasets['training_dataset']
validation_datataset = datasets['validation_dataset']
testing_datataset = datasets['testing_dataset']


print('Number of NODES features: ', training_datataset.num_features)
print('Number of EDGES features: ', training_datataset.num_edge_features)

finish_time_preprocessing = time.time()
time_preprocessing = (finish_time_preprocessing - start_time) / 60 #TODO


# Dataset Split Percent:
''' training_percentage = 0.98
validation_percentage = 0.01
test_percentage = 0.01

n_train = int(len(training_datataset) * training_percentage)
n_val = int(len(training_datataset) * validation_percentage)
n_test = len(training_datataset) - n_train - n_val '''

# Define objetos de conjunto de entrenamiento, validación y prueba de PyTorch:
''' train_set, val_set, test_set = torch.utils.data.random_split(training_datataset,
                                                             [n_train, n_val, n_test],
                                                             generator=torch.Generator().manual_seed(42))
 '''
# Define dataloaders para conjuntos de entrenamiento, validación y prueba:
batch_size = 100  

dataloader = DataLoader(training_datataset, batch_size, shuffle=True)

train_dataloader = DataLoader(training_datataset, batch_size, shuffle=True)
val_dataloader = DataLoader(validation_datataset , batch_size, shuffle=True)
test_dataloader = DataLoader(testing_datataset, batch_size, shuffle=True)

## RUN TRAINING LOOP: 

# Train with a random seed to initialize weights:
torch.manual_seed(0)

# Set up model:
# Initial Inputs
initial_dim_gcn = training_datataset.num_features
edge_dim_feature = training_datataset.num_edge_features

hidden_dim_nn_1 = 20
hidden_dim_nn_2 = 10

hidden_dim_gat_0 = 15

hidden_dim_fcn_1 = 10
hidden_dim_fcn_2 = 5
hidden_dim_fcn_3 = 3 


model = GCN_Geo(
                initial_dim_gcn,
                edge_dim_feature,
                hidden_dim_nn_1,
                hidden_dim_nn_2,

                hidden_dim_gat_0,
                
                hidden_dim_fcn_1,
                hidden_dim_fcn_2,
                hidden_dim_fcn_3,
            ).to(device)

# /////////////////// Transfer Learning /////////////////////////
# Especifica la ruta del archivo donde se guardaron los pesos del modelo previamente
#weights_path = 'SCX_best_model_weights.pth'

# Carga los pesos en tu modelo
#model.load_state_dict(torch.load(weights_path))


#/////////////////// Training /////////////////////////////
# Set up optimizer:
learning_rate = 1E-3 
weight_decay = 1E-5 
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# Definir el scheduler ReduceLROnPlateau
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold= 0.1, verbose= True, mode='max', patience=100, factor=0.1)


train_losses = []
val_losses = []

best_val_loss = float('inf')  # infinito

start_time_training = time.time()
number_of_epochs = 100

for epoch in range(1, number_of_epochs+1):
    train_loss = train(model, device, train_dataloader, optimizer, epoch, type_dataset='training')
    train_losses.append(train_loss)

    val_loss = validation(model, device, val_dataloader, epoch, type_dataset='validation')
    val_losses.append(val_loss)

    # Programar el LR basado en la pérdida de validación
    #scheduler.step(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "weights/best_model_weights.pth")

finish_time_training = time.time()
time_training = (finish_time_training - start_time_training) / 60


#---------------------------------------//////// Losse curves ///////// ---------------------------------------------------------

plt.plot(train_losses, label='Training loss', color='darkorange') 
plt.plot(val_losses, label='Validation loss', color='seagreen')  

# Agregar texto para la mejor pérdida de validación
best_val_loss_epoch = val_losses.index(best_val_loss)  # Calcular el epoch correspondiente a la mejor pérdida de validación
best_val_loss = best_val_loss*100
# Añadir la época y el mejor valor de pérdida como subtítulo
plt.title('Training and Validation Loss\nAMP Dataset\nBest Validation Loss: Epoch {}, Value {:.4f}'.format(best_val_loss_epoch, best_val_loss), fontsize=17)
# Aumentar el tamaño de la fuente en la leyenda
plt.legend(fontsize=14) 
plt.xlabel('Epochs')
plt.ylabel('Loss')

# Guardar la figura en formato PNG con dpi 216
plt.savefig('results/loss_curve.png', dpi=216)
plt.show()

# Testing:
weights_file = "weights/best_model_weights.pth"
threshold = 0.5


# ------------------------------------////////// Training set /////////////---------------------------------------------------
#TODO hacer un script con una funcion que haga todo esto

training_input, training_target, training_pred, training_pred_csv = predict_test(model, train_dataloader, device, weights_file, threshold, type_dataset='training')

#Saving a CSV file with prediction values
prediction_train_set = {
                        'Sequence':training_input,
                        'Target': training_target.cpu().numpy(),
                        'Prediction':  training_pred_csv
                        }

df = pd.DataFrame(prediction_train_set)
df.to_excel('results/training_set_prediction.xlsx', index=False)

bcm = BinaryConfusionMatrix(task="binary", threshold=threshold, num_classes=2).to(device) 
confusion_matrix = bcm(training_pred, training_target)
confusion_matrix_np = confusion_matrix.detach().cpu().numpy()


TN_training = confusion_matrix[0, 0].cpu().numpy()
FP_training = confusion_matrix[0, 1].cpu().numpy()
FN_training = confusion_matrix[1, 0].cpu().numpy()
TP_training = confusion_matrix[1, 1].cpu().numpy()


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
plt.savefig('results/training_bcm.png', dpi=216)
plt.show()

# Cálculo de métricas de evaluación
ACC_training = (TP_training + TN_training) / (TP_training + TN_training + FP_training + FN_training)

PR_training = TP_training / (TP_training + FP_training)

SN_training = TP_training / (TP_training + FN_training)

SP_training = TN_training / (TN_training + FP_training)

F1_training = 2 * (PR_training * SN_training) / (PR_training + SN_training)

# Calculate Matthews Correlation Coefficient (MCC)
mcc_training = (TP_training * TN_training - FP_training * FN_training) / \
                        ((TP_training + FP_training) * (TP_training + FN_training) * \
                         (TN_training + FP_training) * (TN_training + FN_training)) ** 0.5


# Calcular la curva ROC
fpr, tpr, thresholds = roc_curve( training_target.cpu().numpy(), training_pred.cpu().numpy())

# Calcular el área bajo la curva ROC (AUC)
roc_auc_training = auc(fpr, tpr)

# Trazar la curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc_training:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve - Trainning Set')
plt.legend(loc='lower right', fontsize=16)
plt.savefig('results/training_ROC.png', dpi=216)
plt.show()

# Imprimir las métricas
print('/// Evaluation Metrics - Trainning ///\n') 
print(f"Accuracy: {ACC_training:.3f}")
print(f"Precision: {PR_training:.3f}")
print(f"Recall: {SN_training:.3f}")
print(f"Specificity: {SP_training:.3f}")
print(f"F1 Score: {F1_training:.3f}")

#-------------------------------------------- ////////// Validation Set //////////-------------------------------------------------
validation_input, validation_target, validation_pred, validation_pred_csv = predict_test(model, val_dataloader, device, weights_file, threshold, type_dataset='validation')

#Saving a CSV file with prediction values
prediction_validation_set = {
                            'Sequence':validation_input,
                            'Target': validation_target.cpu().numpy(),
                            'Prediction':  validation_pred_csv
                            }

df = pd.DataFrame(prediction_validation_set)
df.to_excel('results/validation_set_prediction.xlsx', index=False)


bcm = BinaryConfusionMatrix(task="binary", threshold=threshold, num_classes=2).to(device) 
confusion_matrix = bcm(validation_pred, validation_target)
confusion_matrix_np = confusion_matrix.detach().cpu().numpy()

TN_validation = confusion_matrix[0, 0].cpu().numpy() 
FP_validation = confusion_matrix[0, 1].cpu().numpy()
FN_validation = confusion_matrix[1, 0].cpu().numpy() 
TP_validation = confusion_matrix[1, 1].cpu().numpy()

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
plt.savefig('results/validation_bcm.png', dpi=216)
plt.show() 

# Cálculo de métricas de evaluación
ACC_validation = (TP_validation + TN_validation) / (TP_validation + TN_validation + FP_validation + FN_validation)

PR_validation = TP_validation / (TP_validation + FP_validation)

SN_validation = TP_validation / (TP_validation + FN_validation)

SP_validation = TN_validation / (TN_validation + FP_validation)

F1_validation = 2 * (PR_validation * SN_validation) / (PR_validation + SN_validation)

#Calculate Matthews Correlation Coefficient (MCC)
mcc_validation = (TP_validation * TN_validation - FP_validation * FN_validation) / \
                        ((TP_validation + FP_validation) * (TP_validation + FN_validation) * \
                         (TN_validation + FP_validation) * (TN_validation + FN_validation)) ** 0.5

# Calcular la curva ROC
fpr, tpr, thresholds = roc_curve( validation_target.cpu().numpy(), validation_pred.cpu().numpy())

# Calcular el área bajo la curva ROC (AUC)
roc_auc_validation = auc(fpr, tpr)

# Trazar la curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc_validation:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve - Validation Set')
plt.legend(loc='lower right', fontsize=16)
plt.savefig('results/validation_ROC.png', dpi=216)
plt.show()

# Imprimir las métricas
print('/// Evaluation Metrics - Validation ///\n') 
print(f"Accuracy: {ACC_validation:.3f}")
print(f"Precision: {PR_validation:.3f}")
print(f"Recall: {SN_validation:.3f}")
print(f"Specificity: {SP_validation:.3f}")
print(f"F1 Score: {F1_validation:.3f}")

# --------------------------------------------////////// Test Set //////////---------------------------------------------------
test_input, test_target, test_pred, test_pred_csv = predict_test(model, test_dataloader, device, weights_file,threshold, type_dataset='testing')

#Saving a CSV file with prediction values
prediction_test_set = {
                        'Sequence':test_input,
                        'Target': test_target.cpu().numpy(),
                        'Prediction': test_pred_csv
                        }

df = pd.DataFrame(prediction_test_set)
df.to_excel('results/test_set_prediction.xlsx', index=False)

bcm = BinaryConfusionMatrix(task="binary",threshold=threshold, num_classes=2).to(device) 
confusion_matrix = bcm(test_pred, test_target)
confusion_matrix_np = confusion_matrix.detach().cpu().numpy()


TN_test = confusion_matrix[0, 0].cpu().numpy()
FP_test = confusion_matrix[0, 1].cpu().numpy()
FN_test= confusion_matrix[1, 0].cpu().numpy()
TP_test = confusion_matrix[1, 1].cpu().numpy()


# Añadir números a la matriz de confusión
cmap = plt.get_cmap('YlGn')
plt.matshow(confusion_matrix_np, cmap=cmap)
plt.title('Confusion Matrix Plot - Test')
plt.colorbar()

# Añadir números a la matriz de confusión
for i in range(confusion_matrix_np.shape[0]):
    for j in range(confusion_matrix_np.shape[1]):
        plt.text(j, i, str(confusion_matrix_np[i, j]), ha='center', va='center', color='black', fontsize=18)

plt.xlabel('Predicted Negative         Predicted Positive')
plt.ylabel('True Positive               True Negative')
plt.savefig('results/test_bcm.png', dpi=216)
plt.show()

# Cálculo de métricas de evaluación
ACC_test = (TP_test + TN_test) / (TP_test + TN_test + FP_test+ FN_test)

PR_test = TP_test / (TP_test + FP_test)

SN_test = TP_test / (TP_test + FN_test)

SP_test = TN_test / (TN_test + FP_test)

F1_test = 2 * (PR_test * SN_test) / (PR_test + SN_test)

#Calculate Matthews Correlation Coefficient (MCC)
mcc_test = (TP_test * TN_test - FP_test * FN_test) / \
                        ((TP_test + FP_test) * (TP_test + FN_test) * \
                         (TN_test + FP_test) * (TN_test + FN_test)) ** 0.5

# Calcular la curva ROC
fpr, tpr, thresholds = roc_curve( test_target.cpu().numpy(), test_pred.cpu().numpy())

# Calcular el área bajo la curva ROC (AUC)
roc_auc_test= auc(fpr, tpr)

# Trazar la curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='mediumseagreen', lw=2, label=f'AUC = {roc_auc_test:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
plt.xlabel('False Positive Rate (FPR)',fontsize=14)
plt.ylabel('True Positive Rate (TPR)',fontsize=14)
plt.title('Receiver Operating Characteristic (ROC) Curve - Test Set')
plt.legend(loc='lower right', fontsize=16)
plt.savefig('results/test_ROC.png', dpi=216)
plt.show()

# Imprimir las métricas
print('/// Evaluation Metrics - Test ///\n') 
print(f"Accuracy: {ACC_test:.3f}")
print(f"Precision: {PR_test:.3f}")
print(f"Recall: {SN_test:.3f}")
print(f"Specificity: {SP_test:.3f}")
print(f"F1 Score: {F1_test:.3f}")


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
    "node_features",
    "edge_features",
    "initial_dim_gcn",
    "edge_dim_feature",
    "hidden_dim_nn_1",
    "hidden_dim_nn_2",
    "hidden_dim_nn_3",
    "hidden_dim_gat_0",
    "hidden_dim_fcn_1",
    "hidden_dim_fcn_2",
    "hidden_dim_fcn_3",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "number_of_epochs",
    "threshold",
    "TP_training",
    "TN_training",
    "FP_training",
    "FN_training",
    "ACC_training",
    "PR_training",
    "SN_training",
    "SP_training",
    "F1_training",
    "mcc_training",
    "roc_auc_training",
    "TP_validation",
    "TN_validation",
    "FP_validation",
    "FN_validation",
    "ACC_validation",
    "PR_validation",
    "SN_validation",
    "SP_validation",
    "F1_validation",
    "mcc_validation",
    "roc_auc_validation",
    "TP_test",
    "TN_test",
    "FP_test",
    "FN_test",
    "ACC_test",
    "PR_test",
    "SN_test",
    "SP_test",
    "F1_test",
    "mcc_test",
    "roc_auc_test",
    "time_preprocessing",
    "time_training",
    "time_prediction",
    "total_time"
    ],
    "Value": [
        training_datataset.num_features,
        training_datataset.num_edge_features,
        initial_dim_gcn,
        edge_dim_feature ,
        hidden_dim_nn_1 ,
        hidden_dim_nn_2 ,
        hidden_dim_nn_3,
        hidden_dim_gat_0,
        hidden_dim_fcn_1 ,
        hidden_dim_fcn_2 ,
        hidden_dim_fcn_3 ,
        batch_size,
        learning_rate,
        weight_decay,
        number_of_epochs,
        threshold,
        TP_training,
        TN_training,
        FP_training,
        FN_training,
        ACC_training, 
        PR_training, 
        SN_training, 
        SP_training,
        F1_training,
        mcc_training,
        roc_auc_training,
        TP_validation,
        TN_validation,
        FP_validation,
        FN_validation,
        ACC_validation, 
        PR_validation, 
        SN_validation, 
        SP_validation,
        F1_validation,
        mcc_validation,
        roc_auc_validation,
        TP_test,
        TN_test,
        FP_test,
        FN_test,
        ACC_test, 
        PR_test, 
        SN_test, 
        SP_test,
        F1_test,
        mcc_test,
        roc_auc_test,
        time_preprocessing, 
        time_training,
        time_prediction,
        total_time
    ],
    
}

''' data = {
    "Metric": [
    "node_features",
    "edge_features",
    "initial_dim_gcn",
    "edge_dim_feature",
    "hidden_dim_nn_1",
    "hidden_dim_nn_2",
    "hidden_dim_nn_3",
    "hidden_dim_gat_0",
    "hidden_dim_fcn_1",
    "hidden_dim_fcn_2",
    "hidden_dim_fcn_3",
    "training_percentage*100",
    "validation_percentage*100",
    "test_percentage*100",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "number_of_epochs",
    "threshold",
    "TP_training",
    "TN_training",
    "FP_training",
    "FN_training",
    "ACC_training",
    "PR_training",
    "SN_training",
    "SP_training",
    "F1_training",
    "roc_auc_training",
    "TP_validation",
    "TN_validation",
    "FP_validation",
    "FN_validation",
    "ACC_validation",
    "PR_validation",
    "SN_validation",
    "SP_validation",
    "F1_validation",
    "roc_auc_validation",
    "TP_test",
    "TN_test",
    "FP_test",
    "FN_test",
    "ACC_test",
    "PR_test",
    "SN_test",
    "SP_test",
    "F1_test",
    "roc_auc_test",
    "time_preprocessing",
    "time_training",
    "time_prediction",
    "total_time"
    ],
    "Value": [
        training_datataset.num_features,
        training_datataset.num_edge_features,
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
        TP_training,
        TN_training,
        FP_training,
        FN_training,
        ACC_training, 
        PR_training, 
        SN_training, 
        SP_training,
        F1_training,
        roc_auc_training,
        TP_validation,
        TN_validation,
        FP_validation,
        FN_validation,
        ACC_validation, 
        PR_validation, 
        SN_validation, 
        SP_validation,
        F1_validation,
        roc_auc_validation,
        TP_test,
        TN_test,
        FP_test,
        FN_test,
        ACC_test, 
        PR_test, 
        SN_test, 
        SP_test,
        F1_test,
        roc_auc_test,
        time_preprocessing, 
        time_training,
        time_prediction,
        total_time
    ],
    
}
 '''

df = pd.DataFrame(data)
df.to_csv('results/results_training_validation_test.csv', index=False)

#------------------Testing new datasets--------------------------------------------

''' 
torch.cuda.empty_cache()
dataloader_test = DataLoader(datasets[test_dataset], batch_size, shuffle=True)
independent_test_input, independent_test_target, independet_test_pred, independet_test_pred_csv = predict_test(model, dataloader_test, device, weights_file, threshold, test_dataset)


#Saving a CSV file with prediction values
indenpendet_test = {
    'Sequence': independent_test_input,
    'Target': independent_test_target.cpu().numpy(),
    'Prediction': independet_test_pred_csv 
}

df_test = pd.DataFrame(indenpendet_test)
df_test.to_excel('results/independet_test.xlsx', index=False)

bcm = BinaryConfusionMatrix(task="binary",threshold=threshold, num_classes=2).to(device) 
confusion_matrix = bcm(independet_test_pred, independent_test_target)
confusion_matrix_np = confusion_matrix.detach().cpu().numpy()

TN_independent_test = confusion_matrix[0, 0].cpu().numpy()
FP_independent_test = confusion_matrix[0, 1].cpu().numpy()
FN_independent_test= confusion_matrix[1, 0].cpu().numpy()
TP_independent_test = confusion_matrix[1, 1].cpu().numpy()


# Añadir números a la matriz de confusión
cmap = plt.get_cmap('Blues')
plt.matshow(confusion_matrix_np, cmap=cmap)
plt.title('Confusion Matrix Plot - Independent dataset - Test')
plt.colorbar()

# Añadir números a la matriz de confusión
for i in range(confusion_matrix_np.shape[0]):
    for j in range(confusion_matrix_np.shape[1]):
        plt.text(j, i, str(confusion_matrix_np[i, j]), ha='center', va='center', color='grey', fontsize=18)

plt.xlabel('Predicted Negative         Predicted Positive')
plt.ylabel('True Positive               True Negative')
plt.savefig('results/independent_bcm.png', dpi=216)
plt.show()

# Cálculo de métricas de evaluación
ACC_independent_test = (TP_independent_test + TN_independent_test) / (TP_independent_test + TN_independent_test + FP_independent_test+ FN_independent_test)

PR_independent_test = TP_independent_test / (TP_independent_test + FP_independent_test)

SN_independent_test = TP_independent_test / (TP_independent_test + FN_independent_test)

SP_independent_test = TN_independent_test / (TN_independent_test + FP_independent_test)

F1_independent_test = 2 * (PR_independent_test * SN_independent_test) / (PR_independent_test + SN_independent_test)

# Calculate Matthews Correlation Coefficient (MCC)
mcc_independent_test = (TP_independent_test * TN_independent_test - FP_independent_test * FN_independent_test) / \
                        ((TP_independent_test + FP_independent_test) * (TP_independent_test + FN_independent_test) * \
                         (TN_independent_test + FP_independent_test) * (TN_independent_test + FN_independent_test)) ** 0.5

# Calcular la curva ROC
fpr, tpr, thresholds = roc_curve(independent_test_target.cpu().numpy(), independet_test_pred.cpu().numpy())

# Calcular el área bajo la curva ROC (AUC)
roc_auc_independent_test= auc(fpr, tpr)

# Trazar la curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc_independent_test:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve - Independet Dataset')
plt.legend(loc='lower right', fontsize=16)
plt.savefig('results/independet_test_ROC.png', dpi=216)
plt.show()

# Saving results to CSV
result_dict = {
    'Accuracy': [ACC_independent_test],
    'Precision': [PR_independent_test],
    'Recall': [SN_independent_test],
    'Specificity': [SP_independent_test],
    'F1 Score': [F1_independent_test],
    'MCC': [mcc_independent_test],
    'AUC-ROC':[roc_auc_independent_test]
}

result_df = pd.DataFrame(result_dict)
result_df.to_csv('results/result_independent_dataset.csv', index=False)

# Imprimir las métricas

print('/// Evaluation Metrics - Test ///\n') 
print(f"Accuracy: {ACC_independent_test:.3f}")
print(f"Precision: {PR_independent_test:.3f}")
print(f"Recall: {SN_independent_test:.3f}")
print(f"Specificity: {SP_independent_test:.3f}")
print(f"F1 Score: {F1_independent_test:.3f}")
print(f"MCC: {mcc_independent_test:.3f}")
 '''

#-------------------------------------///////// Shapley Values Sampling ////// -------------------------------------------
''' 
explainer = Explainer(
    model=model,
    algorithm=CaptumExplainer('ShapleyValueSampling'),
    explanation_type='model',
    model_config=dict(
        mode='binary_classification',
        task_level='node',
        return_type='raw',
    ),
    node_mask_type='attributes',
    edge_mask_type=None,
    threshold_config=dict(
        threshold_type='hard',
        value=0,
    ),
)

# Generar explicaciones para cada nodo en cada lote del DataLoader
explanations = []
aminoacids_features_dict = torch.load('data/dictionaries/train_val_dataset/aminoacids_features_dict.pt')
peptides_features_dict = torch.load('data/dictionaries/train_val_dataset/peptides_features_dict.pt')
blosum62_dict = torch.load('data/dictionaries/train_val_dataset/blosum62_dict.pt')

for batch in test_dataloader:
    batch = batch.to(device)
    x, edge_index,  edge_attr, idx_batch, cc, monomer_labels, num_graphs, target = batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.cc, batch.monomer_labels, batch.num_graphs, batch.y

    explanation = explainer(
                            x=x,
                            target = target,
                            index = None,
                            edge_index=edge_index,
                            edge_attr=edge_attr,
                            aminoacids_features_dict=aminoacids_features_dict,
                            peptides_features_dict=peptides_features_dict,
                            blosum62_dict=blosum62_dict,
                            idx_batch=idx_batch,
                            cc=cc,
                            monomer_labels=monomer_labels,
                            num_graphs=num_graphs
                        )


# Visualizar la importancia de las características para cada nodo en cada lote
for i, explanation in enumerate(explanations):
    path = f'results/feature_importance_node_{i}.png'
    explanation.visualize_feature_importance(path)
    print(f"Feature importance plot for node {i} has been saved to '{path}'")
 '''
# %%
