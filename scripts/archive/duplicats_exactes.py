import hashlib
import os

import pandas as pd


# Defineix la ruta de la carpeta que vols analitzar
dataset_path = "/home/edxnG11/dataset_wikiart/raw/"

# Extensions d'imatge acceptades
extensions_imatge = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff")

# Fitxer on es guardara el detall dels duplicats trobats
fitxer_sortida = "duplicats_exactes.csv"


def calcular_hash_fitxer(ruta_fitxer, mida_bloc=1024 * 1024):
    sha256 = hashlib.sha256()

    with open(ruta_fitxer, "rb") as fitxer:
        for bloc in iter(lambda: fitxer.read(mida_bloc), b""):
            sha256.update(bloc)

    return sha256.hexdigest()


rutes_imatges = []

for arrel, _, fitxers in os.walk(dataset_path):
    for nom_fitxer in fitxers:
        if nom_fitxer.lower().endswith(extensions_imatge):
            rutes_imatges.append(os.path.join(arrel, nom_fitxer))

total_imatges = len(rutes_imatges)
print(f"S'han trobat {total_imatges} imatges. Calculant hashes...", flush=True)

imatges_per_hash = {}

for index, ruta_imatge in enumerate(rutes_imatges, start=1):
    try:
        hash_imatge = calcular_hash_fitxer(ruta_imatge)
        imatges_per_hash.setdefault(hash_imatge, []).append(ruta_imatge)
    except Exception as error:
        print(f"No s'ha pogut llegir la imatge {ruta_imatge}: {error}", flush=True)

    if index % 1000 == 0 or index == total_imatges:
        print(f"Processades {index}/{total_imatges} imatges...", flush=True)

grups_duplicats = {
    hash_imatge: rutes
    for hash_imatge, rutes in imatges_per_hash.items()
    if len(rutes) > 1
}

total_grups = len(grups_duplicats)
total_imatges_duplicades = sum(len(rutes) for rutes in grups_duplicats.values())
total_duplicats_extra = sum(len(rutes) - 1 for rutes in grups_duplicats.values())

print("\nResum de duplicats exactes:")
print(f"- Total d'imatges analitzades: {total_imatges}")
print(f"- Grups de duplicats exactes: {total_grups}")
print(f"- Imatges dins de grups duplicats: {total_imatges_duplicades}")
print(f"- Duplicats extra respecte a una copia original: {total_duplicats_extra}")
print("-" * 60)

if not grups_duplicats:
    print("No s'han trobat imatges duplicades exactes.")
else:
    files_csv = []

    for numero_grup, (hash_imatge, rutes) in enumerate(grups_duplicats.items(), start=1):
        print(f"\nGrup {numero_grup} - {len(rutes)} imatges duplicades")
        print(f"Hash SHA256: {hash_imatge}")

        for ruta in rutes:
            print(f"  - {ruta}")
            files_csv.append({
                "Grup": numero_grup,
                "Hash_SHA256": hash_imatge,
                "Ruta": ruta,
            })

    df_duplicats = pd.DataFrame(files_csv)
    df_duplicats.to_csv(fitxer_sortida, index=False)
    print(f"\nDetall dels duplicats guardat a: {fitxer_sortida}")
