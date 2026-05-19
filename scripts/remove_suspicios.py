import os
import sys
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from utils.data_utils import get_cached_image_path

CSV_PATH = "scripts/results/analysis/suspicious_images_report.csv"

SOURCE_ROOT = "/home/datasets/wikiart"
CACHE_ROOT = "/tmp/wikiart_336"

OUTPUT_PATH = "scripts/results/analysis/exclude_suspicious_paths.txt"

df = pd.read_csv(CSV_PATH)
suspicious = df[df["suspicion_score"] > 0].copy()

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for original_path in suspicious["path"]:
        cache_path = get_cached_image_path(SOURCE_ROOT, CACHE_ROOT, original_path)
        f.write(cache_path + "\n")

print(f"Imatges cachejades a excloure: {len(suspicious)}")
print(f"Llista guardada a: {OUTPUT_PATH}")