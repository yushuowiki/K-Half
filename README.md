# K-Half

## Requirements

Ensure you have the following dependencies installed:

```bash
pip install torch==11.3 transformers numpy scikit-learn matplotlib tqdm
```

## Usage

### Labeling the Dataset

```bash
python label.py --dataset <DATASET_NAME>
```

Example:

```bash
python label.py --dataset ICEWS14
```

### Generating Output

```bash
python out.py --dataset <DATASET_NAME> --train_file <train|test|valid> --threshold <VALIDITY_THRESHOLD>
```

Example:

```bash
python out.py --dataset ICEWS14 --train_file train --threshold 0.01
```

### Training the Model

```bash
python train.py --dataset <DATASET_NAME> --train_file <train|test|valid> --entity_out_dim_1 <DIM> --entity_out_dim_2 <DIM> --epochs <NUM_EPOCHS> --batch <BATCH_SIZE> --threshold <THRESHOLD>
```

Example:

```bash
python train.py --dataset ICEWS14 --train_file train --entity_out_dim_1 32 --entity_out_dim_2 32 --epochs 50 --batch 5000 --threshold 0
```

## Dataset Preparation

Navigate to the `data` directory and transfer the dataset to the model you want to run:

```bash
cd data
# Place your dataset files here
```

Ensure your dataset is formatted correctly before running the model.
