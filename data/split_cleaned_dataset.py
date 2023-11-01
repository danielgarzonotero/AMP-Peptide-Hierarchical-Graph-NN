
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
target_rows = 100

if total_rows > target_rows:
    
    rows_to_drop = total_rows - target_rows

    
    rows_to_drop_indices = random.sample(range(total_rows), rows_to_drop)

    df = df.drop(rows_to_drop_indices)

df.to_csv('100_all_AMP_suffled.csv', index=False, quoting=None)
df = pd.read_csv('100_all_AMP_suffled.csv')
filas, columnas = df.shape
print(f"El DataFrame tiene {filas} filas y {columnas} columnas.")


# %%
