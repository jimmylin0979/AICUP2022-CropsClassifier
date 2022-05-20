# AICUP2022 CropsClassifier

## Environment setup

```
git clone https://github.com/jimmylin0979/AICUP2022-CropsClassifier.git
cd AICUP2022-CropsClassifier

pip install -r requirements.txt
```

## Getting Start

Note that the default location of the Crops dataset training and testing sets in the configuration file are `./data/dataset/train` and `./data/dataset/test` respectively. Please make changes accordingly.

Bt the way, you can alter batch size, initial learning rate or num of epochs in the configuration file. For more infomation, please have a look at `config.py`

### Training

```
python main.py
```

### Testing

The final prediction on Crops testing dataset is 0.9802984, 48/151.

```
python evaluate.py
```
