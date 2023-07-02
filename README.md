# Customized ViT

## Installation
Install the conda environment:
```
conda env create -f environment.yml
conda activate vit
```

## Specify your Config
Modify the dictionary in /configs/main.py and execute to generate a .yml file:
```
python3.8 /configs/main.py
```
Modify /shells/train_vit_example.sh to specify your training settings.

## Get started
Train the model by your modified config, or existing configs:
```
sh /shells/train_vit_example.sh
```
or
```
sh /shells/train_vit_default.sh
```
