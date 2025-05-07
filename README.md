# AI-for-siRNA-Effect-Prediction（第二界世界AI4S大赛 11名/2200+队伍 ）

## Project Overview
This project aims to predict the effects of siRNA drugs for the World Second AI4Science Competition. The code and environment setup described here are intended to ensure that results can be reproduced consistently.

## System Requirements
- **Operating System:** Ubuntu 20.04
- **Python Version:** 3.9.17
- **Pytorch Version:** 1.12.1
- **viennarna Version:** 2.6.4
- **cudNN Version:** cudnn7.6.5_0
- **CUDA Version:** 10.2
- **GPU:** NVIDIA TITAN RTX

## Environment Setup
The environment dependencies are listed in the `environment.yaml` file. You can create the Conda environment using the following command:

```bash
conda env create -f environment.yaml
```

### Environment Dependencies
The `environment.yaml` file includes:
- **Conda Channels:** `pytorch`, `defaults`
- **Dependencies:** Various packages including `pytorch` 1.12.1 with CUDA 10.2, `viennarna==2.6.4`,`numpy==1.24.4`, `pandas==1.5.3`, `scikit-learn==1.3.0`, and others.

The complete list of dependencies is detailed in the `environment.yaml` file, which includes package versions and additional pip-installed packages.

## File Structure
```
project
|-- README.md
|-- sirna_prediction_environment.yaml
|-- data
|   |-- external_data
|   |   |-- readme.md
|   |   |-- train_data_aug3.2.csv
|   |   |-- sample_submission_aug3.2.csv
|-- code
|   |-- main.py
|-- submit
|   |-- submit_20240822.csv
```

## Explanation of External Datasets
- Regarding the external dataset, we did not use additional samples; instead, we calculated some derived features based on the existing samples, such as the binding affinity between siRNA and target mRNA.

## Random Seed
- To ensure reproducibility, random seeds are set in the code. 
