# Repo installation

```
conda create -p ./.env python=3.9 -y
conda activate ./.env 
conda install pytorch torchtext torchvision -c pytorch -y
conda install poetry -y
poetry install
```