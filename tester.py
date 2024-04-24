#%%
import torch
import pandas as pd
from torch_geometric.loader import DataLoader
from src.data import GeoDataset
from src.process import predict_test
from src.model import GCN_Geo
from src.device import device_info
from src.evaluation_metrics import evaluate_model

device_information = device_info()
device = device_information.device
batch_size = 100
threshold = 0.5

# Cargar los datos de prueba
indep_testing_dataset = GeoDataset(raw_name='data/dataset/Med_School_1.csv')
indep_testing_dataloader = DataLoader(indep_testing_dataset, batch_size, shuffle=False)

# Set up model:
# Initial Inputs
initial_dim_gcn = indep_testing_dataset.num_features
edge_dim_feature = indep_testing_dataset.num_edge_features

hidden_dim_nn_1 = 20
hidden_dim_nn_2 = 10

hidden_dim_gat_0 = 10

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

weights_file="weights/best_model_weights_1.pth"

# Ejecutar la función de predicción en el conjunto de datos de prueba utilizando el modelo cargado
indep_testing_input, indep_testing_target, indep_testing_pred, indep_testing_pred_csv, indep_testing_scores = predict_test(model, indep_testing_dataloader, device, weights_file, threshold, type_dataset='testing')

# Guardar un archivo CSV con los valores de predicción
indep_prediction_test_set = {
    'Sequence': indep_testing_input,
    'Target': indep_testing_target.cpu().numpy().T.flatten().tolist(),
    'Scores' : indep_testing_scores,
    'Prediction': indep_testing_pred_csv, 
}

df = pd.DataFrame(indep_prediction_test_set)
df.to_excel('results/indep_testing_prediction.xlsx', index=False)

# Evaluar el rendimiento del modelo en los datos de prueba
TP_indep_testing, TN_indep_testing, FP_indep_testing, FN_indep_testing, ACC_indep_testing, PR_indep_testing, \
SN_indep_testing, SP_indep_testing, F1_indep_testing, mcc_indep_testing, roc_auc_indep_testing = \
evaluate_model(prediction=indep_testing_pred,
               target=indep_testing_target,
               dataset_type='Testing',
               threshold=threshold,
               device=device)


# %%
