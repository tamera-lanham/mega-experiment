# Repo installation

First make sure you have [conda installed](https://docs.conda.io/en/latest/miniconda.html).

```
conda create -p ./.env python=3.9 -y
conda activate ./.env 
conda install pytorch torchtext torchvision -c pytorch -y
conda install poetry -y
poetry install
```

Whenever you start a new terminal, run `conda activate ./.env` to activate the virtual environment where the project is installed.