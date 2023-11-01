
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
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
from math import sqrt
from torchmetrics.classification import BinaryConfusionMatrix

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


# # Number of datapoints in the training set:
training_test_percentage = 0.70 #TODO
n_train = int(len(dataset) * training_test_percentage)

# # Number of datapoints in the validation set:
n_val = len(dataset) - n_train

# # Define pytorch training and validation set objects:
train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

# # Build pytorch training and validation set dataloaders:
batch_size = 25 #TODO
dataloader = DataLoader(dataset, batch_size, shuffle=True)
train_dataloader = DataLoader(train_set, batch_size, shuffle=True)
val_dataloader = DataLoader(val_set, batch_size, shuffle=True)

## RUN TRAINING LOOP: 

# Train with a random seed to initialize weights:
torch.manual_seed(0)

# Set up model:
# Initial Inputs
initial_dim_gcn = dataset.num_features
edge_dim_feature = dataset.num_edge_features

hidden_dim_nn_1 = 500
hidden_dim_nn_2 = 250   
hidden_dim_nn_3 = 100

hidden_dim_gat_0 = 15

hidden_dim_fcn_1 = 100
hidden_dim_fcn_2 = 50
hidden_dim_fcn_3 = 10


model = GCN_Geo(
                initial_dim_gcn,
                edge_dim_feature,
                hidden_dim_nn_1,
                hidden_dim_nn_2,
                hidden_dim_nn_3,

                hidden_dim_gat_0,

                hidden_dim_fcn_1,
                hidden_dim_fcn_2,
                hidden_dim_fcn_3
            ).to(device)


# Set up optimizer:
learning_rate = 1E-5
optimizer = optim.Adam(model.parameters(), learning_rate)

train_losses = []
val_losses = []

best_val_loss = float('inf')  # infinito

start_time_training = time.time()
number_of_epochs = 300
for epoch in range(1, number_of_epochs):
    train_loss = train(model, device, train_dataloader, optimizer, epoch)
    train_losses.append(train_loss)

    val_loss = validation(model, device, val_dataloader, epoch)
    val_losses.append(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        
        torch.save(model.state_dict(), "best_model_weights.pth")
     
finish_time_training = time.time()
time_training = (finish_time_training - start_time_training) / 60

#Testing:
weights_file = "best_model_weights.pth"

# Training:
input_all, target_all_train, pred_prob_all_train = predict_test(model, train_dataloader, device, weights_file)

bcm = BinaryConfusionMatrix(task="binary", num_classes=2).to(device)
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
        plt.text(j, i, str(confusion_matrix_np[i, j]), ha='center', va='center', color='black')

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
print('///Evaluation Metrics - Trainning///\n') #TODO incluir en el csv
print(f"Accuracy: {accuracy_train:.3f}")
print(f"Precision: {precision_train:.3f}")
print(f"Recall: {recall_train:.3f}")
print(f"Specificity: {specificity_train:.3f}")
print(f"F1 Score: {f1_score_train:.3f}")

# Validation:

input_all, target_all_validation, pred_prob_all_validation = predict_test(model, val_dataloader, device, weights_file)

bcm = BinaryConfusionMatrix(task="binary", num_classes=2).to(device)
confusion_matrix = bcm(pred_prob_all_validation, target_all_validation)
confusion_matrix_np = confusion_matrix.detach().cpu().numpy()

true_positives_validation = confusion_matrix[1, 1].cpu().numpy()
true_negatives_validation = confusion_matrix[0, 0].cpu().numpy() 
false_positives_validation = confusion_matrix[0, 1].cpu().numpy()
false_negatives_validation = confusion_matrix[1, 0].cpu().numpy() 

cmap = plt.get_cmap('Blues')
plt.matshow(confusion_matrix_np, cmap=cmap)
plt.title('Confusion Matrix Plot - Validation')
plt.colorbar()

# Añadir números a la matriz de confusión
for i in range(confusion_matrix_np.shape[0]):
    for j in range(confusion_matrix_np.shape[1]):
        plt.text(j, i, str(confusion_matrix_np[i, j]), ha='center', va='center', color='black')

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
print('///Evaluation Metrics - Validation///\n') #TODO incluir en el csv
print(f"Accuracy: {accuracy_val:.3f}")
print(f"Precision: {precision_val:.3f}")
print(f"Recall: {recall_val:.3f}")
print(f"Specificity: {specificity_val:.3f}")
print(f"F1 Score: {f1_score_val:.3f}")

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


#Times
finish_time = time.time()
time_prediction = (finish_time - finish_time_training) / 60
total_time = (finish_time - start_time) / 60
print("\n //// Preprocessing time: {:3f} minutes ////".format(time_preprocessing))
print("\n //// Training time: {:3f} minutes ////".format(time_training))
print("\n //// Prediction time: {:3f} minutes ////".format(time_prediction))
print("\n //// Total time: {:3f} minutes ////".format(total_time))

# Result DataFrame
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
        "training_test_percentage %",
        "batch_size", 
        "learning_rate",
        "number_of_epochs",
        "true_positives_train",
        "true_negatives_train",
        "false_positives_train",
        "false_negatives_train",
        "true_positives_validation",
        "true_negatives_validation",
        "false_positives_validation",
        "false_negatives_validation",
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
        training_test_percentage*100,
        batch_size,
        learning_rate,
        number_of_epochs,
        true_positives_train,
        true_negatives_train,
        false_positives_train,
        false_negatives_train,
        true_positives_validation,
        true_negatives_validation,
        false_positives_validation,
        false_negatives_validation,
        time_preprocessing, 
        time_training,
        time_prediction,
        total_time
    ],
    
}


df = pd.DataFrame(data)
df.to_csv('results/results.csv', index=False)



# %%
