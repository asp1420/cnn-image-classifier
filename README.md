# Image Classifier

This code enables the creation of a model that classifies images based on a specified data structure.


## 🛠️ Config

To execute the code set the following variables in the `fit.yaml` file.

- `trainer:logger:save_dir`: Output path for saving model weights.
- `model:num_classes`: Total number of classes.
- `model:learning_rate`: Learning rate for the optimizer.
- `model:network`: Type of network/model to use: `INCEPTION`, `RESNET50`, `RESNEXT50`, `EFFICIENTNETV2`, `SWINGTRANSFORMERV2`.
- `model:pretrained`: Indicates whether to use pretrained weights.
- `data:path`: Path to the data, following the structure outlined in the Data Structure section.
- `data:input_size`: Shape used for the input image, e.g., 256 for 256x256.
- `data:batch_size`: Batch size used during the training process.
- `data:workers`: Number of CPU cores allocated for the data loader.

## 🗂️ Data structure

The data structure consists of two main directories: `train` and `validation`. Each of these directories contains a set of subdirectories named with consecutive numbers (0, 1, ..., N), where the numbers correspond to the total number of classes.

```
dataset
├── train
│   ├── 0
│   ├── 1
│   ...
│   └── N
├── validation
|   ├── 0
|   ├── 1
|   ...
│   └── N
```

## ⚙️ Usage

To execute training process run the following in console/terminal:

```bash
python main.py fit --config fit.yaml
```
