import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


# Defineix la ruta de la carpeta que vols analitzar
dataset_path = "/home/edxnG11/dataset_wikiart/raw/"

# Nombre d'imatges extremes que es mostraran per pantalla
num_extrems = 10

# Extensions d'imatge acceptades
extensions_imatge = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff")

resolucions = []
rutes_imatges = []

for arrel, _, fitxers in os.walk(dataset_path):
    for nom_fitxer in fitxers:
        if nom_fitxer.lower().endswith(extensions_imatge):
            rutes_imatges.append(os.path.join(arrel, nom_fitxer))

total_imatges = len(rutes_imatges)
print(f"S'han trobat {total_imatges} imatges. Comencant l'analisi...", flush=True)

for index, ruta_imatge in enumerate(rutes_imatges, start=1):
    try:
        with Image.open(ruta_imatge) as img:
            amplada, alcada = img.size

        pixels_totals = amplada * alcada
        resolucions.append({
            "Ruta": ruta_imatge,
            "Amplada": amplada,
            "Alcada": alcada,
            "Pixels": pixels_totals,
            "Resolucio": f"{amplada}x{alcada}",
        })
    except Exception as error:
        print(f"No s'ha pogut llegir la imatge {ruta_imatge}: {error}", flush=True)

    if index % 1000 == 0 or index == total_imatges:
        print(f"Processades {index}/{total_imatges} imatges...", flush=True)

df_resolucions = pd.DataFrame(resolucions)

if df_resolucions.empty:
    print("No s'han trobat imatges per analitzar les resolucions.")
else:
    df_resolucions = df_resolucions.sort_values(by="Pixels")

    imatges_mes_petites = df_resolucions.head(num_extrems)
    imatges_mes_grans = df_resolucions.tail(num_extrems).sort_values(
        by="Pixels",
        ascending=False,
    )

    print("Resum de resolucions:")
    print(f"- Total d'imatges analitzades: {len(df_resolucions)}")
    print(f"- Resolucio minima: {df_resolucions.iloc[0]['Resolucio']}")
    print(f"- Resolucio maxima: {df_resolucions.iloc[-1]['Resolucio']}")
    print("-" * 30)

    print("\nImatges amb resolucions mes petites:")
    print(imatges_mes_petites[["Resolucio", "Pixels", "Ruta"]].to_string(index=False))

    print("\nImatges amb resolucions mes grans:")
    print(imatges_mes_grans[["Resolucio", "Pixels", "Ruta"]].to_string(index=False))

    plt.figure(figsize=(10, 6))
    plt.hist(df_resolucions["Pixels"], bins=50, color="mediumseagreen", edgecolor="black")
    plt.title("Histograma de resolucions de les imatges", fontsize=16)
    plt.xlabel("Resolucio total (amplada x alcada, en pixels)", fontsize=12)
    plt.ylabel("Nombre d'imatges", fontsize=12)
    plt.tight_layout()

    nom_fitxer = "histograma_resolucions.png"
    plt.savefig(nom_fitxer, dpi=300)
    print(f"\nHistograma guardat correctament com a: {nom_fitxer}")
