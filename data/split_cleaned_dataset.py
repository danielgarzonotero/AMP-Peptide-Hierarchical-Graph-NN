
#%%
import pandas as pd
import random

df = pd.read_csv('datasets/test_nonamp.fasta' )

df.to_csv('datasets/Siu_test_nonamp.csv', index=False, quoting=None)

df = pd.read_csv('datasets/Siu_test_nonamp.csv')
filas, columnas = df.shape
print(f"El DataFrame tiene {filas} filas y {columnas} columnas.")


# %%
import pandas as pd


# Lee el archivo CSV en un DataFrame
df = pd.read_csv('Xiao_AMP_trainc09n3g2d1.fasta', header=None)

# Filtra las filas que contienen secuencias de aminoácidos
df = df[~(df.index % 2 == 0)]
# Resetea los índices del DataFrame
df = df.reset_index(drop=True)

# Guarda el DataFrame resultante en un nuevo archivo CSV
df.to_csv('Xiao_AMP_trainc09n3g2d1.csv', header=False, index=False)
filas, columnas = df.shape
print(f"El DataFrame tiene {filas} filas y {columnas} columnas.")


#%%
import pandas as pd

# Lee el archivo CSV en un DataFrame
df = pd.read_csv('Xiao_AMP_trainc04n2g2d1.csv', header=None)

# Agrega una columna con el valor 1 a cada fila
df['Activity'] = 1

# Guarda el DataFrame resultante en un nuevo archivo CSV con un tabulador como delimitador
df.to_csv('Xiao_AMP_trainc04n2g2d1.csv', sep=',', header=None, index=False)
filas, columnas = df.shape
print(f"El DataFrame tiene {filas} filas y {columnas} columnas.")


# %% /////////////// Unir dos datasets ///////////////////////
import pandas as pd

# Lee los dos DataFrames desde archivos CSV sin encabezados
df1 = pd.read_csv('Xiao_AMP_trainc09n3g2d1.csv')
print(df1.shape)
df2 = pd.read_csv('Xiao_nonAMP_09train.csv')
print(df2.shape)

# Combina los DataFrames sin tener en cuenta los encabezados
df_combined = pd.concat([df1, df2], ignore_index=True)

# Renombra las columnas como "Sequence" y "Activity"
df_combined.columns = ['Sequence', 'Activity']

# Guarda el DataFrame combinado en un archivo CSV
df_combined.to_csv('datasets/09_Xiao_training.csv', index=False)

# Imprime la forma del DataFrame combinado
filas, columnas = df_combined.shape
print(f"El DataFrame tiene {filas} filas y {columnas} columnas.")


# %% //////// Revolver un dataset aleatoriamente ////////////////////

import pandas as pd
import random

df = pd.read_csv('datasets/04_Xiao_training.csv' )


df_shuffled = df.sample(frac=1, random_state=1).reset_index(drop=True)
df_shuffled.to_csv('datasets/04_Xiao_training.csv', index=False, quoting=None)

df = pd.read_csv('datasets/04_Xiao_training.csv')
filas, columnas = df.shape
print(f"El DataFrame tiene {filas} filas y {columnas} columnas.")


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

#%%
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
    df_presentes.to_csv('comparation/presentes.csv', index=False)
    print('The number of Sequences in the Jing Xu et al. dataset that are in the Chia-Ru Chung et al. dataset is: ', len(df_presentes.iloc[:, 0]))
    df_no_presentes.to_csv('comparation/no_presentes.csv', index=False)
    print('The number of Sequences in the Jing Xu et al. dataset that are NOT in the Chia-Ru Chung et al. dataset is: ', len(df_no_presentes.iloc[:, 0]))

# Ejemplo de uso
buscar_y_crear_datasets('datasets/Jing_all_amp_nonamp_suffled.csv', 'datasets/Chia_shuffled_amp_nonamp_train_and_test.csv')


# %% //////////////// Para revolever aleatoriamente un dataset ////////// 
import pandas as pd

# Lee el CSV
df = pd.read_csv('Xiao_training.csv')

# Revuelve las filas aleatoriamente
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Guarda el DataFrame revuelto en un nuevo CSV
df_shuffled.to_csv('Xiao_training.csv', index=False)

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


# %%//////////////////// Extraer columnas ///////////////////////////////////


import pandas as pd

# Ruta del archivo CSV de entrada
archivo_entrada = 'datasets/Uniandes_Train.csv'

# Ruta del nuevo archivo CSV de salida
archivo_salida = 'datasets/Uniandes_Train.csv'

# Leer el archivo CSV de entrada
datos = pd.read_csv(archivo_entrada)

# Seleccionar las columnas 1 y 2
columnas_seleccionadas = datos.iloc[:, [0, 1]]

# Guardar las columnas seleccionadas en un nuevo archivo CSV
columnas_seleccionadas.to_csv(archivo_salida, index=False)

print(f'Nuevo archivo CSV creado en: {archivo_salida}')

# %% //////// Encontrar duplicados /////////////
import csv

def find_duplicate_first_column(csv_file):
    duplicates = set()
    seen = set()
    
    with open(csv_file, 'r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] in seen:
                duplicates.add(row[0])
            else:
                seen.add(row[0])
    
    return duplicates

# Reemplaza 'tu_archivo.csv' con el nombre de tu archivo CSV
csv_file = 'datasets/Xiao_nonamp_all_suffled_noduplicates.csv'
duplicates = find_duplicate_first_column(csv_file)

if duplicates:
    print("Duplicados encontrados en la primera columna:")
    for value in duplicates:
        print(value)
else:
    print("No se encontraron duplicados en la primera columna.")

# %%//////// Elimnar duplicados /////////////
import csv

def remove_duplicates_and_count(csv_file):
    duplicates_count = 0
    unique_values = set()
    
    # Lista para almacenar filas únicas
    unique_rows = []
    
    with open(csv_file, 'r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0] not in unique_values:
                unique_values.add(row[0])
                unique_rows.append(row)
            else:
                duplicates_count += 1
    
    # Escribir filas únicas en un nuevo archivo CSV
    with open('datasets/Xiao_amp_all_suffled_noduplicates.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(unique_rows)
    
    return duplicates_count

# Reemplaza 'tu_archivo.csv' con el nombre de tu archivo CSV
csv_file = 'datasets/Xiao_amp_all_suffled.csv'
duplicates_count = remove_duplicates_and_count(csv_file)

print(f"Se encontraron {duplicates_count} duplicados en la primera columna.")
print("Los duplicados han sido eliminados. Se ha guardado el resultado en 'sin_duplicados.csv'.")

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
# %%
