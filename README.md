# ShaSpec - CVPR2023

The official code repository of ShaSpec model from CVPR 2023 [paper](https://arxiv.org/pdf/2307.14126) "Multi-modal Learning with Missing Modality via Shared-Specific Feature Modelling"

## Installation

```commandline
pip install -r requirements.txt
```

For more requirements, please refer to requirements.txt

## Data Preparation

BraTS2018 dataset has 285 cases for training/validation (210 gliomas with high grade and 75 gliomas with low grade) and 66 cases for online evaluation, where each case (with four modalities, namely: Flair, T1, T1CE and T2) share one segmentation GT. The ground-truth of training set is publicly available, but the annotations of validation set is hidden and online evaluation is required.

The data can be requested from [here](https://www.kaggle.com/datasets/sanglequang/brats2018).

The data path can be changed in `datalist/BraTS18/`. There are 4 files in the folder: `BraTS18_train.csv` and `BraTS18_val.csv` for hyper-params tuning; `BraTS18_train_all.csv` for fixed iteration training with all data; and `BraTS18_test.csv` for online evaluation at [here](https://ipp.cbica.upenn.edu/).

## Model Training

Followed the official BraTS2018 settings, the models are trained on training data for a certain iterations and then tested on online evaluation data. Detailed hyper-parameters settings can be found in `run.sh` and in the paper. Note that we empirically found out a lower temperature of random modality dropout can help at the initial stage of the training as the model performance is not stable and gradually increase the dropout rate. Alternatively, we can perform a warmup with all modalities training as shown in the run.sh script.

For model training, the commandline is:

```commandline
bash run.sh [GPU id]
```

For instance:

```commandline
bash run.sh 0
```

## Model Evaluation

For model evaluation, the resume path of the tested model can be specified in the `eval.sh` file. The evaluation can be performed with:

```commandline
bash eval.sh [GPU id]
```

For example:

```commandline
bash eval.sh 0
```

Then if you want to perform postprocessing, please run:

```commandline
python postprocess.py
```

The folder paths can be modified in `postprocess.py`.

After postprocessing, [online evaluation](https://ipp.cbica.upenn.edu/) needed to be performed. Output folder containing 66 segmentations is required to be uploaded to the site for evaluation.

Enjoy!!
