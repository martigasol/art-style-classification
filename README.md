# WikiArt Style Classification

Projecte de classificació d'estils artístics amb WikiArt. El model principal és una ResNet50 preentrenada a ImageNet i adaptada al nombre de classes del nostre problema.

## Estructura

```text
main.py                         # entrenament principal
train.py                        # bucle train/validation i checkpoints
test.py                         # avaluació final sobre test
models/transfer_model.py        # ResNet50 amb transfer learning
utils/data_utils.py             # càrrega, split, transforms i balanceig
utils/dataset.py                # Dataset PyTorch
scripts/check_ood_duplicates.py # duplicats exactes/visuals OOD vs WikiArt
scripts/evaluate_ood_folder_exp27.py # avaluació OOD del checkpoint EXP27
results/                        # checkpoints, figures i mètriques
```

Els scripts antics que no formen part del flux final són a `scripts/archive/`.

## Dependències

```bash
conda env create -f environment.yml
conda activate wikiart-classification
```

## Dataset

El dataset WikiArt s'espera organitzat en carpetes, una per estil:

```text
/home/datasets/wikiart/
    Impressionism/
    Realism/
    ...
```

El dataset extern OOD també va per carpetes i usa les 27 classes originals:

```text
data/ood_art_external/
    Abstract_Expressionism/
    Action_painting/
    ...
```

## Entrenament

`main.py` conté la config de l'experiment principal EXP27: ResNet50, 27 classes, imatge 384, fine-tuning parcial, class weights, label smoothing i scheduler per validation macro F1.

```bash
python main.py
```

Per reproduir variants explicades a la presentació, canvia només els flags de la config: `use_top_k_classes`, `use_class_grouping`, `feature_extraction`, `partial_finetuning`, `unfreeze_layer3`, `use_weighted_sampler` o el mode de cache d'aspect ratio.

Per defecte:

```python
use_top_k_classes = False
use_class_grouping = False
class_groups = {}
```

## Avaluació OOD

Aquest script només fa inferència sobre el dataset extern. No entrena, no agrupa classes i no aplica top-k.

```bash
python scripts/evaluate_ood_folder_exp27.py \
  --ood_root data/ood_art_external \
  --checkpoint_path results/checkpoints/exp27_resnet50_384_all_classes_best.pth \
  --image_size 384 \
  --batch_size 16 \
  --num_workers 4 \
  --output_dir results/ood_exp27
```

Resultats:

```text
results/ood_exp27/ood_predictions.csv
results/ood_exp27/ood_summary.csv
results/ood_exp27/ood_confusion_matrix.csv
results/ood_exp27/ood_classification_report.txt
```

## Duplicats OOD

Comprova duplicats exactes i possibles duplicats visuals entre WikiArt i l'OOD. No elimina fitxers.

```bash
python scripts/check_ood_duplicates.py \
  --original_root /home/datasets/wikiart \
  --ood_root data/ood_art_external \
  --output_dir results/ood_duplicate_check \
  --visual_threshold 5
```

Resultats:

```text
results/ood_duplicate_check/exact_duplicates.csv
results/ood_duplicate_check/visual_duplicates.csv
results/ood_duplicate_check/ood_class_distribution_without_visual_duplicates.csv
results/ood_duplicate_check/summary.txt
```

## Experiments principals

- Top-14 classes per reduir desbalanceig inicial.
- Comparació amb les 27 classes originals.
- Feature extraction vs fine-tuning parcial.
- Descongelar més capes amb `unfreeze_layer3`.
- Class weights i weighted sampler per tractar el desbalanceig.
- Augment de resolució a 384.
- Preprocessing amb resize quadrat i proves d'aspect ratio.
- Agrupació de classes només quan `use_class_grouping=True`.
- Avaluació OOD amb un mini dataset extern.
