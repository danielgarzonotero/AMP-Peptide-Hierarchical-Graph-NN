#%%^
import pandas as pd
import matplotlib.pyplot as plt

# AMP Dataset
df_amp = pd.read_csv('AMP_labeled.csv', names=['sequence', 'activity'])

# Add a column with the length of each sequence
df_amp['len'] = df_amp['sequence'].apply(len)

# Calculate mean and standard deviation
mean_amp = df_amp['len'].mean()
std_amp = df_amp['len'].std()
df_amp.to_csv('AMP_analisis.csv', sep=',', index=False)

# Histogram of sequence lengths
values_amp, edges_amp, _ = plt.hist(df_amp['len'], bins=100, color="g") 
plt.xlabel('Sequence Length')
plt.ylabel('Frequency')
plt.title('Distribution of AMP Sequence Lengths')

# Add legend with mean and standard deviation
plt.text(0.95, 0.85, f"Mean: {mean_amp:.2f}\nStd: {std_amp:.2f}", 
         transform=plt.gca().transAxes, ha='right', color='black',
         bbox=dict(facecolor='white', edgecolor='grey', boxstyle='round,pad=0.5'))

plt.show()

# ------------------------------Non-AMP Dataset------------------------------
df_nonamp = pd.read_csv('nonAMP_labeled.csv', names=['sequence', 'activity'])

# Add a column with the length of each sequence
df_nonamp['len'] = df_nonamp['sequence'].apply(len)

# Calculate mean and standard deviation
mean_nonamp = df_nonamp['len'].mean()
std_nonamp = df_nonamp['len'].std()
df_nonamp.to_csv('nonAMP_analisis.csv', sep=',', index=False)

# Histogram of sequence lengths
values_nonamp, edges_nonamp, _ = plt.hist(df_nonamp['len'], bins=100, color="firebrick") 
plt.xlabel('Sequence Length')
plt.ylabel('Frequency')
plt.title('Distribution of Non-AMP Sequence Lengths')

# Add legend with mean and standard deviation
plt.text(0.95, 0.85, f"Mean: {mean_nonamp:.2f}\nStd: {std_nonamp:.2f}", 
         transform=plt.gca().transAxes, ha='right', color='black',
         bbox=dict(facecolor='white', edgecolor='grey', boxstyle='round,pad=0.5'))

plt.show()

# %%
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import table

# Lista de los 20 aminoácidos
aminoacidos = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
# ------------------------------AMP Dataset------------------------------
df_amp = pd.read_csv('AMP_analisis.csv')

# Crear columnas para cada aminoácido y contar su frecuencia en cada secuencia
for aa in aminoacidos:
    df_amp[aa] = df_amp['sequence'].apply(lambda x: x.count(aa))

df_amp.to_csv('AMP_analisis.csv', sep=',', index=False)

# ------------------------------NOnAMP Dataset------------------------------
df_namp = pd.read_csv('nonAMP_analisis.csv')

# Crear columnas para cada aminoácido y contar su frecuencia en cada secuencia
for aa in aminoacidos:
    df_namp[aa] = df_namp['sequence'].apply(lambda x: x.count(aa))

df_namp.to_csv('nonAMP_analisis.csv', sep=',', index=False)

#----------------------------Distributions-------------------------------------

def plot_aminoacid_distribution(df, dataset_name, aminoacidos, color):
    # Crear un DataFrame para facilitar el trazado
    df_plot = df[aminoacidos]

    # Calcular el promedio y la desviación estándar para cada aminoácido
    means = df_plot.mean().round(4)
    stds = df_plot.std().round(4)

    # Encontrar los 5 aminoácidos con la mayor y menor media
    top5_means = means.nlargest(5)
    bottom5_means = means.nsmallest(5)
    
    # Ordenar los datos de mayor a menor
    top5_means = top5_means.sort_values(ascending=False)
    bottom5_means = bottom5_means.sort_values(ascending=False)

    # Trazar gráfico de barras con barras de error
    plt.bar(aminoacidos, means, yerr=stds, capsize=5, alpha=0.7, color=color)

    # Añadir tabla con los 5 aminoácidos con mayor media
    table_data_top5 = pd.DataFrame(top5_means, columns=['Top 5 Max Mean'])
    table(ax=plt.gca(), data=table_data_top5, loc='bottom', bbox=[0, -0.65, 0.3, 0.4])

    # Añadir tabla con los 5 aminoácidos con menor media
    table_data_bottom5 = pd.DataFrame(bottom5_means, columns=['Bottom 5 Min Mean'])
    table(ax=plt.gca(), data=table_data_bottom5, loc='bottom', bbox=[0.7, -0.65, 0.3, 0.4])

    plt.xlabel('Amino Acid')
    plt.ylabel('Mean')
    plt.title(f'Amino Acid Mean({dataset_name})')
    plt.show()

# AMP Dataset
df_amp = pd.read_csv('AMP_analisis.csv')
plot_aminoacid_distribution(df_amp, 'AMP', aminoacidos, 'g')

# Non-AMP Dataset
df_nonamp = pd.read_csv('nonAMP_analisis.csv')
plot_aminoacid_distribution(df_nonamp, 'Non-AMP', aminoacidos, 'firebrick')







# %%
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import table

# Lista de los 20 aminoácidos
aminoacidos = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
# ------------------------------AMP Dataset------------------------------
df_amp = pd.read_csv('AMP_analisis.csv')

# Crear columnas para cada aminoácido y contar su frecuencia en cada secuencia
for aa in aminoacidos:
    df_amp[aa] = df_amp['sequence'].apply(lambda x: x.count(aa))/df_amp['sequence'].apply(len)

df_amp.to_csv('AMP_analisis2.csv', sep=',', index=False)

# ------------------------------NOnAMP Dataset------------------------------
df_namp = pd.read_csv('nonAMP_analisis.csv')

# Crear columnas para cada aminoácido y contar su frecuencia en cada secuencia
for aa in aminoacidos:
    df_namp[aa] = df_namp['sequence'].apply(lambda x: x.count(aa))/df_amp['sequence'].apply(len)

df_namp.to_csv('nonAMP_analisis2.csv', sep=',', index=False)

#----------------------------Distributions-------------------------------------

def plot_aminoacid_distribution(df, dataset_name, aminoacidos, color):
    # Crear un DataFrame para facilitar el trazado
    df_plot = df[aminoacidos]

    # Calcular el promedio y la desviación estándar para cada aminoácido
    means = df_plot.mean().round(4)
    stds = df_plot.std().round(4)

    # Encontrar los 5 aminoácidos con la mayor y menor media
    top5_means = means.nlargest(5)
    bottom5_means = means.nsmallest(5)
    # Ordenar los datos de mayor a menor
    top5_means = top5_means.sort_values(ascending=False)
    bottom5_means = bottom5_means.sort_values(ascending=False)

    # Trazar gráfico de barras con barras de error
    plt.bar(aminoacidos, means, yerr=stds, capsize=5, alpha=0.7, color=color)

    # Añadir tabla con los 5 aminoácidos con mayor media
    table_data_top5 = pd.DataFrame(top5_means, columns=['Top 5 Max Mean'])
    table(ax=plt.gca(), data=table_data_top5, loc='bottom', bbox=[0, -0.65, 0.3, 0.4])

    # Añadir tabla con los 5 aminoácidos con menor media
    table_data_bottom5 = pd.DataFrame(bottom5_means, columns=['Bottom 5 Min Mean'])
    table(ax=plt.gca(), data=table_data_bottom5, loc='bottom', bbox=[0.7, -0.65, 0.3, 0.4])

    plt.xlabel('Amino Acid')
    plt.ylabel('Mean/len(Sequence)')
    plt.title(f'Amino Acid Mean/len(Sequence) ({dataset_name})')
    plt.show()

# AMP Dataset
df_amp = pd.read_csv('AMP_analisis2.csv')
plot_aminoacid_distribution(df_amp, 'AMP', aminoacidos, 'g')

# Non-AMP Dataset
df_nonamp = pd.read_csv('nonAMP_analisis2.csv')
plot_aminoacid_distribution(df_nonamp, 'Non-AMP', aminoacidos, 'firebrick')
# %%
