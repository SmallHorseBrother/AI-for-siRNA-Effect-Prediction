# AI-for-siRNA-Effect-Prediction

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
