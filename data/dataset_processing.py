
#%% /////////////// Comberting 1 fasta file of amp or nonamp //////////////////////
import pandas as pd

def fasta_processing(path, activity, path_saving):
    df = pd.read_csv(path, header=None)
    
    df = df[~(df.index % 2 == 0)]
    df = df.reset_index(drop=True)

    df['Activity'] = activity
    
    #Checking:
    filas, columnas = df.shape
    print(f"-amp dataset has {filas} rows and {columnas} columns.")
    
    # Renombra las columnas como "Sequence" y "Activity"
    df.columns = ['Sequence', 'Activity']
    
    df.to_csv(path_saving, index=False, quoting=None)

    return

path = 'dataset Siu/Siu_test_amp.fasta'
path_saving= 'TESTING.csv'
activity = 1
fasta_processing(path, activity, path_saving)

#%% /////////////// Combining two fasta files of amp and nonamp //////////////////////
import pandas as pd

def fasta_processing(path_amp, path_nonamp, path_saving):
    df_amp = pd.read_csv(path_amp, header=None)
    df_nonamp = pd.read_csv(path_nonamp, header=None)
    
    df_amp = df_amp[~(df_amp.index % 2 == 0)]
    df_amp = df_amp.reset_index(drop=True)

    df_nonamp = df_nonamp[~(df_nonamp.index % 2 == 0)]
    df_nonamp = df_nonamp.reset_index(drop=True)
    
    df_amp['Activity'] = 1
    df_nonamp['Activity'] = 0
    
    #Checking:
    filas, columnas = df_amp.shape
    print(f"\n-amp dataset has {filas} rows and {columnas} columns.")
    filas, columnas = df_nonamp.shape
    print(f"-nonamp dataset has {filas} rows and {columnas} columns.")
    
    df_combined = pd.concat([df_amp, df_nonamp], ignore_index=True)

    # Renombra las columnas como "Sequence" y "Activity"
    df_combined.columns = ['Sequence', 'Activity']
    
    df_shuffled = df_combined.sample(frac=1, random_state=1).reset_index(drop=True)
    filas, columnas = df_shuffled .shape
    print(f"-The datasets combined have {filas} rows and {columnas} columns.")
    
    df_shuffled.to_csv(path_saving, index=False, quoting=None)

    return

path_amp = 'dataset_Chung/fasta/Chung_1593_all_training_c09n3g2.fasta'
path_nonamp= 'datasets_Xiao/Xiao_nonAMP_train.fasta'
path_saving= 'Chung_Xiao_balanced_all_training.csv'
fasta_processing(path_amp, path_nonamp, path_saving)

path_amp = 'dataset_Chung/fasta/Chung_454_all_validation_c09n3g2.fasta'
path_nonamp= 'datasets_Xiao/Xiao_nonAMP_validation.fasta'
path_saving= 'Chung_Xiao_balanced_all_validation.csv'
fasta_processing(path_amp, path_nonamp, path_saving)

path_amp = 'dataset_Chung/fasta/Chung_226_all_test_c09n3g2.fasta'
path_nonamp= 'datasets_Xiao/Xiao_nonAMP_test.fasta'
path_saving= 'Chung_Xiao_balanced_all_testing.csv'
fasta_processing(path_amp, path_nonamp, path_saving)

# %% ///////////// csv to fasta //////////////////
import csv

def convert_to_fasta(input_csv, output_fasta):
    with open(input_csv, 'r', newline='') as csv_file, open(output_fasta, 'w') as fasta_file:
        reader = csv.reader(csv_file)
        next(reader)  # Saltar la primera fila que contiene los encabezados

        i = 1
        for row in reader:
            sequence = row[0]
            fasta_file.write(f">P{i}\n{sequence}\n")
            i += 1

# Reemplaza 'entrada.csv' y 'salida.fasta' con los nombres de tu archivo CSV de entrada y archivo FASTA de salida, respectivamente
convert_to_fasta('90_Xiao_AMP_train.csv', '90_Xiao_AMP_train.fasta')
convert_to_fasta('90_Xiao_nonAMP_train.csv', '90_Xiao_nonAMP_train.fasta')
print("El archivo CSV se ha convertido a formato FASTA exitosamente.")


#%% /////////////// Extraer un numero X de filas aleatoriamente //////////////////
import pandas as pd
import random

df = pd.read_csv('Xiao_nonAMP_train.csv' )
df_target = pd.read_csv('Xiao_AMP_trainc09n3g2d1.csv' )

total_rows = len(df)

target_rows = len(df_target)

if total_rows > target_rows:
    
    rows_to_drop = total_rows - target_rows

    
    rows_to_drop_indices = random.sample(range(total_rows), rows_to_drop)

    df = df.drop(rows_to_drop_indices)

df.to_csv('Xiao_nonAMP_09train.csv', index=False, quoting=None)
df = pd.read_csv('Xiao_nonAMP_09train.csv')
filas, columnas = df.shape
print(f"El DataFrame tiene {filas} filas y {columnas} columnas.")

#%% ////////////// Removing duplicates ////////////////////
import pandas as pd

# Lee el archivo CSV
df = pd.read_csv('datasets/Xiao_AMP_train_final_3.csv')

# Imprime la cantidad de duplicados encontrados
cantidad_duplicados = df.duplicated(subset=[df.columns[0]]).sum()
print(f'Se encontraron {cantidad_duplicados} duplicados.')

# Elimina las filas duplicadas basadas en el valor de la columna 1
df_sin_duplicados = df.drop_duplicates(subset=[df.columns[0]])

# Guarda el DataFrame resultante en un nuevo archivo CSV
df_sin_duplicados.to_csv('datasets/Xiao_AMP_train_final_4.csv', index=False)


# %% //////////// Histogram /////////////
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('SCX.csv')
print(df.shape)
RT_values = df.iloc[:, 1]  

# Imprimir el promedio de RT
mean_RT = RT_values.mean()
print(f"Mean RT: {mean_RT} minutes")

# Distribution Dataset
plt.hist(RT_values, bins=10)  
plt.title("SCX RT Distribution")
plt.xlabel("RT Values (min)")
plt.ylabel("Frequency")
plt.show()

# %% /////// Para filtrar y guardar from a csv file con varias columnas //////////
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def filter_and_save(path_dataset, condition, output_csv):
    df = pd.read_csv(path_dataset)

    if condition == 'amp':
        condition_filter = ((df['Activity'] == 1) )
    elif condition == 'nonamp':
        condition_filter = ((df['Activity'] == 0) )

    filtered_df = df[condition_filter].copy()

    # Select only the 'sequence' and 'antibacterial' columns
    selected_columns = ['Sequence', 'Activity']
    filtered_df = filtered_df[selected_columns]

    # Save the filtered DataFrame to a new CSV file
    filtered_df.to_csv(output_csv, index=False)

# Example usage:
filter_and_save('datasets/Jing+Chia_without_duplicados.csv', 'nonamp', 'datasets/Jing+Chia_all_nonamp.csv')

# %% ////// Para buscar secuencias presentes en un dataset en otro //////////
import pandas as pd

def buscar_y_crear_datasets(dataset_principal, dataset_busqueda):
    
    df_principal = pd.read_csv(dataset_principal)
    df_busqueda = pd.read_csv(dataset_busqueda)

    presentes = []
    no_presentes = []

    # Iterar sobre las secuencias del dataset principal
    for secuencia in df_principal.iloc[:, 0]:  # Utilizar iloc para seleccionar la primera columna por posición
        # Verificar si la secuencia está presente en el dataset de búsqueda
        if secuencia in df_busqueda.iloc[:, 0].values:  # Utilizar iloc para seleccionar la primera columna por posición
            presentes.append(secuencia)
        else:
            no_presentes.append(secuencia)

    # Crear datasets con las secuencias presentes y no presentes
    df_presentes = df_principal[df_principal.iloc[:, 0].isin(presentes)]  # Utilizar iloc para seleccionar la primera columna por posición
    df_no_presentes = df_principal[df_principal.iloc[:, 0].isin(no_presentes)]  # Utilizar iloc para seleccionar la primera columna por posición

    # Guardar los nuevos datasets en archivos CSV numerados
    df_presentes.to_csv('comparation/presentes.csv', index=False)
    print('The number of Sequences in the Jing Xu et al. dataset that are in the Chia-Ru Chung et al. dataset is: ', len(df_presentes.iloc[:, 0]))
    df_no_presentes.to_csv('comparation/no_presentes.csv', index=False)
    print('The number of Sequences in the Jing Xu et al. dataset that are NOT in the Chia-Ru Chung et al. dataset is: ', len(df_no_presentes.iloc[:, 0]))

# Ejemplo de uso
buscar_y_crear_datasets('datasets/Jing_all_amp_nonamp_suffled.csv', 'datasets/Chia_shuffled_amp_nonamp_train_and_test.csv')

# %% ////////////// Eliminar secuencias de un dataset en otro ///////////

import pandas as pd

# Lee los conjuntos de datos desde archivos CSV
df2 = pd.read_csv('presentes.csv')  # Reemplaza 'dataset1.csv' con el nombre de tu primer archivo CSV
filas, columnas = df2.shape
print(f"El DataFrame tiene {filas} filas y {columnas} columnas.")

df1 = pd.read_csv('datasets/Jing_all_amp_nonamp_suffled.csv')  # Reemplaza 'dataset2.csv' con el nombre de tu segundo archivo CSV
filas, columnas = df1.shape
print(f"El DataFrame tiene {filas} filas y {columnas} columnas.")

# Elimina filas de df1 que tienen IDs en común con df2
result_df = df1[~df1['sequence'].isin(df2['sequence'])]
filas, columnas = result_df.shape
print(f"El DataFrame tiene {filas} filas y {columnas} columnas.")

# Guarda el nuevo conjunto de datos en un nuevo archivo CSV
result_df.to_csv('Jing_all_amp_nonamp_duplicated_removed.csv', index=False)  # Reemplaza 'nuevo_dataset.csv' con el nombre que desees para el nuevo archivo

