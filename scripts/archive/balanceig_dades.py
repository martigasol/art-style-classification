import os
import pandas as pd
import matplotlib.pyplot as plt

# 1. Defineix la ruta on tens descomprimit el dataset
dataset_path = "/home/datasets/wikiart/" 

# Diccionari per guardar { 'Nom_Estil' : quantitat_imatges }
recompte_classes = {}

# 2. Recorrem les carpetes
for nom_carpeta in os.listdir(dataset_path):
    ruta_carpeta = os.path.join(dataset_path, nom_carpeta)
    
    # Ens assegurem que és una carpeta i no un fitxer solt
    if os.path.isdir(ruta_carpeta):
        # Comptem quants fitxers (imatges) hi ha dins
        num_imatges = len(os.listdir(ruta_carpeta))
        recompte_classes[nom_carpeta] = num_imatges

# 3. Ho passem a un DataFrame de Pandas per ordenar-ho fàcilment
df = pd.DataFrame(list(recompte_classes.items()), columns=['Estil', 'Num_Imatges'])

# Ordenem de la classe amb més imatges a la que en té menys
df = df.sort_values(by='Num_Imatges', ascending=False)

# 4. Imprimim el resum numèric
total_imatges = df['Num_Imatges'].sum()
total_classes = len(df)
print(f"Resum del Dataset:")
print(f"- Total de classes (Estils): {total_classes}")
print(f"- Total d'imatges: {total_imatges}")
print("\nDistribució per classe (de més a menys imatges):")
for index, row in df.iterrows():
    print(f"{row['Estil']}: {row['Num_Imatges']} imatges")
print("-" * 30)

# 5. Creem el gràfic visual (el que posareu a la presentació)
plt.figure(figsize=(12, 6))
plt.bar(df['Estil'], df['Num_Imatges'], color='skyblue', edgecolor='black')

plt.title("Distribució d'imatges per Estil (WikiArt)", fontsize=16)
plt.xlabel("Estil Pictòric", fontsize=12)
plt.ylabel("Nombre d'Imatges", fontsize=12)

# Girem els noms de l'eix X perquè es puguin llegir bé
plt.xticks(rotation=45, ha='right') 

# Ajustem marges i mostrem
plt.tight_layout()
nom_fitxer = "distribucio_estils.png"
plt.savefig(nom_fitxer, dpi=300) 
print(f"Gràfic guardat correctament com a: {nom_fitxer}")