# Titanic tutorials

[link](https://www.kaggle.com/c/titanic)

## environments

```bash
conda create -n my_kaggle python=3.8.5 pip
conda activate
conda activate my_kaggle
pip install kaggle
pip install numpy matplotlib pandas
```

- dataset download

```
kaggle competitions download -c titanic
```

- submission

```
kaggle competitions submit -c titanic -f titanic/out.csv -m "Message"
```
