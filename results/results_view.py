#%%

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import pandas as pd
import matplotlib.pyplot as plt

def distribution(property, path_dataset, condition):
    # 1: Length
    # 2: Charge
    # 3: Amphiphilicity
    # 4: Hydropathy
    # 5: Secondary Structure
    # 6: FAI
    # 7: Molecular Weight
    # 8: Hydrophobicity
    # 9: Aromaticity
    # 10: Isoelectric Point
    # 11: Instability Index
    
    #verdaderos: all values well predicted (tp, tn)
    #falsos: all values not predicted (fp, fn)
    #tp: true positive
    #tn: true negative
    #fp: false positive
    #fn: false negative
    
    #Filtering the result excel depending the condition
    df = pd.read_excel(path_dataset)

    if condition == 'verdaderos':
        condition_filter = ((df['Target'] == 0) & (df['Prediction'] == 0)) | ((df['Target'] == 1) & (df['Prediction'] == 1))
    elif condition == 'falsos':
        condition_filter = ((df['Target'] == 1) & (df['Prediction'] == 0)) | ((df['Target'] == 0) & (df['Prediction'] == 1))
    elif condition =='tp':
        condition_filter = (df['Target'] == 1) & (df['Prediction'] == 1)
    elif condition =='tn':
        condition_filter = (df['Target'] == 0) & (df['Prediction'] == 0)
    elif condition =='fp':
        condition_filter = (df['Target'] == 0) & (df['Prediction'] == 1)
    elif condition =='fn':
        condition_filter = (df['Target'] == 1) & (df['Prediction'] == 0)
    
    # Aplicar el filtro a todo el DataFrame
    df_new = df[condition_filter].copy()       
    
    if property == 1:
        property = 'Length'
        df_new[property] = df_new['Sequence'].apply(len)
        # Calculate mean and standard deviation
        mean = df_new[property].mean()
        std = df_new[property].std()
        pass
        
    elif property == 2:
        property = 'Charge'
        df_new[property] = df_new['Sequence'].apply(lambda x: ProteinAnalysis(x).charge_at_pH(7))
        # Calculate mean and standard deviation
        mean = df_new[property].mean()
        std = df_new[property].std()
        pass
    
    elif property == 3:
        property = 'Amphiphilicity'
        # Leer el DataFrame original con la tabla de valores
        df_valores = pd.read_csv('index.csv')  # Reemplaza con la ruta correcta
        
        # Función para calcular la suma de valores para una secuencia dada
        def calcular_suma(secuencia):
            return sum(df_valores.loc[df_valores['Letter'].isin(list(secuencia)), 'Amphiphilicity'])
        
        # Agregar una nueva columna al DataFrame de secuencias con las sumas calculadas
        df_new[property] = df_new['Sequence'].apply(calcular_suma)
        mean = df_new[property].mean()
        std = df_new[property].std()
        pass
    
    elif property == 4:
        property = 'Hydropathy'
        df_valores = pd.read_csv('index.csv')  
        
        def calcular_suma(secuencia):
            return sum(df_valores.loc[df_valores['Letter'].isin(list(secuencia)), 'Hydropathy'])
    
        df_new[property] = df_new['Sequence'].apply(calcular_suma)
        mean = df_new[property].mean()
        std = df_new[property].std()
        pass
    
    elif property == 5:
        property = 'Secondary Structure'
        
        def predict_secondary_structure(peptide_sequence):
            protein_analysis = ProteinAnalysis(peptide_sequence)
            sheet, turn, helix = protein_analysis.secondary_structure_fraction()
            return sheet, turn, helix
        
        # Añadir columnas con la longitud y la estructura secundaria de cada secuencia
        df_new['helix'] = df_new['Sequence'].apply(lambda x: predict_secondary_structure(x)[2])
        df_new['turn'] = df_new['Sequence'].apply(lambda x: predict_secondary_structure(x)[1])
        df_new['sheet'] = df_new['Sequence'].apply(lambda x: predict_secondary_structure(x)[0])

        # Calcular medias y desviaciones estándar
        helix_mean = df_new['helix'].mean()
        helix_std = df_new['helix'].std()
        turn_mean = df_new['turn'].mean()
        turn_std = df_new['turn'].std()
        sheet_mean = df_new['sheet'].mean()
        sheet_std = df_new['sheet'].std()      
        pass
    
    elif property == 6:
        property = 'FAI'
        def fai(sequence):
            helm = peptide_to_helm(sequence)
            mol = Chem.MolFromHELM(helm)
            num_anillos = rdMolDescriptors.CalcNumRings(mol)
            
            charged_amino_acids = {'R': 2, 'H': 1, 'K': 2}
            cationic_charges = sum(sequence.count(aa) * charge for aa, charge in charged_amino_acids.items())

            # para evitar un error matemático
            if num_anillos == 0:
                return 0

            return (cationic_charges / num_anillos)


        # Convertir a notación HELM para usar RDKit
        def peptide_to_helm(sequence):
            polymer_id = 1
            sequence_helm = "".join(sequence)
            sequence_helm = ''.join([c + '.' if c.isupper() else c for i, c in enumerate(sequence_helm)])
            sequence_helm = sequence_helm.rstrip('.')
            sequence_helm = f"PEPTIDE{polymer_id}{{{sequence_helm}}}$$$$"
            return sequence_helm
        
        df_new[property] = df_new['Sequence'].apply(lambda x: fai(x))
        # Calcular medias y desviaciones estándar
        mean= df_new[property].mean()
        std= df_new[property].std()
        pass
    
    elif property == 7:
        property = 'Molecular Weight'
        
        def mw_peptide(peptide_sequence):
            peptide_analysis = ProteinAnalysis(peptide_sequence)
            mw_peptide = peptide_analysis .molecular_weight()
            return mw_peptide
        
        df_new[property] = df_new['Sequence'].apply(lambda x: mw_peptide(x))
        # Calcular medias y desviaciones estándar
        mean= df_new[property].mean()
        std = df_new[property].std()
        pass
    
    elif property == 8:
        property = 'Hidrophobicity'
        
        def hidrophobicity(peptide_sequence):
            peptide_analysis = ProteinAnalysis(peptide_sequence)
            hidrophobicity = peptide_analysis.gravy()
            return hidrophobicity
        
        df_new[property] = df_new['Sequence'].apply(lambda x: hidrophobicity(x))
        # Calcular medias y desviaciones estándar
        mean= df_new[property].mean()
        std = df_new[property].std()
        pass
    
    elif property == 9:
        property = 'Aromaticity'
        def aromaticity(peptide_sequence):
            peptide_analysis = ProteinAnalysis(peptide_sequence)
            aromaticity = peptide_analysis.aromaticity()
            return aromaticity
        
        df_new[property] = df_new['Sequence'].apply(lambda x: aromaticity(x))
        # Calcular medias y desviaciones estándar
        mean= df_new[property].mean()
        std = df_new[property].std()       
        pass
    
    elif property == 10:
        property = 'Isoelectric Point'
        def isoelectric_point(peptide_sequence):
            peptide_analysis = ProteinAnalysis(peptide_sequence)
            isoelectric_point = peptide_analysis.isoelectric_point()
            return isoelectric_point
        
        df_new[property] = df_new['Sequence'].apply(lambda x: isoelectric_point(x))
        # Calcular medias y desviaciones estándar
        mean= df_new[property].mean()
        std = df_new[property].std()   
        pass
    
    elif property == 11:
        property = 'Instability Index'
        def instability_index(peptide_sequence):
            peptide_analysis = ProteinAnalysis(peptide_sequence)
            instability_index= peptide_analysis.instability_index()
            return instability_index
        
        df_new[property] = df_new['Sequence'].apply(lambda x: instability_index(x))
        # Calcular medias y desviaciones estándar
        mean= df_new[property].mean()
        std = df_new[property].std()         
        pass
        
    else:
       #NEW PROPERTIES TO ADD
        pass

    #Plottig
    color = "green" if condition =='verdaderos' or condition== 'tp' or condition == 'tn' else "red"
    if condition =='verdaderos':
        clase = "True Predicted"
    elif condition =='falsos':
        clase = "False Predicted"
    elif condition =='tp':
        clase = "True Positives"  
    elif condition =='tn':
        clase = "True Negatives"     
    elif condition =='fp':
        clase = "False Positives"
    elif condition =='fn':
        clase = "False Negatives"
    
    if property == 'Secondary Structure':
        plt.figure(figsize=(10, 6))

        plt.hist(df_new['helix'], bins=100, color="g", alpha=0.9, label="Helix") 
        plt.hist(df_new['turn'], bins=100, color="b", alpha=0.5, label="Turn") 
        plt.hist(df_new['sheet'], bins=100, color="r", alpha=0.2, label="Sheet") 

        plt.xlabel('Secondary Structure Fraction ',size= 15)
        plt.ylabel('Frequency',size= 15)
        plt.title(f"Distribution of {property} - {clase}",size= 15)

        # Agregar leyenda
        plt.legend(fontsize='20')

        # Añadir texto con medias y desviaciones estándar
        plt.text(0.95, 0.055, f"Helix: Mean={helix_mean:.2f}, Std={helix_std:.2f}\nTurn: Mean={turn_mean:.2f}, Std={turn_std:.2f}\nSheet: Mean={sheet_mean:.2f}, Std={sheet_std:.2f}", 
                transform=plt.gca().transAxes, ha='right', color='black',
                bbox=dict(facecolor='white', edgecolor='grey', boxstyle='round,pad=0.5'), size= 12)

        plt.show()
        
        
    else:
        plt.figure(figsize=(10, 6))
        plt.hist(df_new[property], bins=100, color=color, alpha=0.9) 
        plt.xlabel(property, size= 15)
        plt.ylabel('Frequency',size= 15)
        plt.title(f"Distribution of property {property} - {clase}",size= 15)

        # Añadir texto con medias y desviaciones estándar
        plt.text(0.95, 0.85, f"Mean={mean:.2f}\nStd={std:.2f}", 
                transform=plt.gca().transAxes, ha='right', color='black',
                bbox=dict(facecolor='white', edgecolor='grey', boxstyle='round,pad=0.5'), size= 14)
        plt.show()



# AMP Dataset
distribution(11, 'prediction_test_set.xlsx', 'tp')

# Non-AMP Dataset
#distribution(1, 'nonAMP_analisis.csv', 'nonamp')
# %%
