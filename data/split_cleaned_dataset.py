
#%%
import pandas as pd
import random

df = pd.read_csv('final_AMPs.fasta' )

df.to_csv('AMP.csv', index=False, quoting=None)

df = pd.read_csv('AMP.csv')
filas, columnas = df.shape
print(f"El DataFrame tiene {filas} filas y {columnas} columnas.")


# %%
import pandas as pd


# Lee el archivo CSV en un DataFrame
df = pd.read_csv('nonAMP.csv', header=None)

# Filtra las filas que contienen secuencias de aminoácidos
df = df[~(df.index % 2 == 0)]
# Resetea los índices del DataFrame
df = df.reset_index(drop=True)

# Guarda el DataFrame resultante en un nuevo archivo CSV
df.to_csv('nonAMP_cleaned.csv', header=False, index=False)
filas, columnas = df.shape
print(f"El DataFrame tiene {filas} filas y {columnas} columnas.")


#%%
import pandas as pd

# Lee el archivo CSV en un DataFrame
df = pd.read_csv('nonAMP_cleaned.csv', header=None)

# Agrega una columna con el valor 1 a cada fila
df['activity'] = 0

# Guarda el DataFrame resultante en un nuevo archivo CSV con un tabulador como delimitador
df.to_csv('nonAMP_labeled.csv', sep=',', header=None, index=False)
filas, columnas = df.shape
print(f"El DataFrame tiene {filas} filas y {columnas} columnas.")

# %%
import pandas as pd

# Lee los dos DataFrames desde archivos CSV
df1 = pd.read_csv('AMP_labeled.csv', header = None)
filas, columnas = df1.shape
print(f"El DataFrame AMP tiene {filas} filas y {columnas} columnas.")

df2 = pd.read_csv('nonAMP_labeled.csv', header = None)
filas, columnas = df2.shape
print(f"El DataFrame nonAMP tiene {filas} filas y {columnas} columnas.")

df_combined = df1.append(df2, ignore_index=True)

df_combined.rename(columns={0: 'sequence', 1: 'activity'}, inplace=True)

filas, columnas = df_combined.shape
print(f"El DataFrame tiene {filas} filas y {columnas} columnas.")
df_combined.to_csv('all_AMP_justappend.csv', index=False)


# %%

import pandas as pd
import random

df = pd.read_csv('all_AMP_justappend.csv' )


df_shuffled = df.sample(frac=1, random_state=1).reset_index(drop=True)
df_shuffled.to_csv('all_AMP_suffled.csv', index=False, quoting=None)

df = pd.read_csv('all_AMP_suffled.csv')
filas, columnas = df.shape
print(f"El DataFrame tiene {filas} filas y {columnas} columnas.")


#%%
import pandas as pd
import random

df = pd.read_csv('all_AMP_suffled.csv' )

total_rows = len(df)
target_rows = 25

if total_rows > target_rows:
    
    rows_to_drop = total_rows - target_rows

    
    rows_to_drop_indices = random.sample(range(total_rows), rows_to_drop)

    df = df.drop(rows_to_drop_indices)

df.to_csv('25_all_AMP_suffled.csv', index=False, quoting=None)
df = pd.read_csv('25_all_AMP_suffled.csv')
filas, columnas = df.shape
print(f"El DataFrame tiene {filas} filas y {columnas} columnas.")


# %% //////////// Histograma /////////////
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
        condition_filter = ((df['Antibacterial'] == 1) )
    elif condition == 'nonamp':
        condition_filter = ((df['Antibacterial'] == 0) )

    filtered_df = df[condition_filter].copy()

    # Select only the 'sequence' and 'antibacterial' columns
    selected_columns = ['Sequence', 'Antibacterial']
    filtered_df = filtered_df[selected_columns]

    # Save the filtered DataFrame to a new CSV file
    filtered_df.to_csv(output_csv, index=False)

# Example usage:
filter_and_save('analisis/comparation/multiAMP_test.csv', 'nonamp', 'nonamp_all_test_filtered.csv')

# %% ////// Para buscar secuencias de un dataset en otro //////////
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
    df_presentes.to_csv('presentes.csv', index=False)
    print('The number of Sequences in the Jing Xu et al. dataset that are in the Chia-Ru Chung et al. dataset is: ', len(df_presentes.iloc[:, 0]))
    df_no_presentes.to_csv('no_presentes.csv', index=False)
    print('The number of Sequences in the Jing Xu et al. dataset that are NOT in the Chia-Ru Chung et al. dataset is: ', len(df_no_presentes.iloc[:, 0]))

# Ejemplo de uso
buscar_y_crear_datasets('datasets/Jing_all_amp_nonamp_suffled.csv', 'datasets/Chia_shuffled_amp_nonamp_train_and_test.csv')


# %% /////////// Para unir dos datasets /////////////
import pandas as pd

# Lee los dos DataFrames desde archivos CSV
df1 = pd.read_csv('Jing_all_amp_nonamp_duplicated_removed.csv', header=None)
filas, columnas = df1.shape
print(f"El DataFrame AMP tiene {filas} filas y {columnas} columnas.")

df2 = pd.read_csv('datasets/Chia_shuffled_amp_nonamp_train_and_test.csv', header=None)
filas, columnas = df2.shape
print(f"El DataFrame nonAMP tiene {filas} filas y {columnas} columnas.")

df_combined = pd.concat([df1, df2])

df_combined.rename(columns={0: 'sequence', 1: 'activity'}, inplace=True)

filas, columnas = df_combined.shape
print(f"El DataFrame tiene {filas} filas y {columnas} columnas.")
df_combined.to_csv('Jing_Chia_without_duplicados.csv', index=False)


# %% //////////////// Para revolever aleatoriamente un dataset ////////// 
import pandas as pd

# Lee el CSV
df = pd.read_csv('amp_nonamp_train_and_test.csv')

# Revuelve las filas aleatoriamente
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Guarda el DataFrame revuelto en un nuevo CSV
df_shuffled.to_csv('shuffled_amp_nonamp_train_and_test.csv', index=False)

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


# %%
