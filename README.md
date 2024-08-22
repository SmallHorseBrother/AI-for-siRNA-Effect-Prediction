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

## Environment Setup
The environment dependencies are listed in the `environment.yaml` file. You can create the Conda environment using the following command:

```bash
conda env create -f environment.yaml
```

### Environment Dependencies
The `environment.yaml` file includes:
- **Conda Channels:** `pytorch`, `defaults`
- **Dependencies:** Various packages including `pytorch` 1.12.1 with CUDA 10.2, `numpy`, `pandas`, `scikit-learn`, and others.

The complete list of dependencies is detailed in the `environment.yaml` file, which includes package versions and additional pip-installed packages.

## File Structure
```
project
|-- README.md
|-- data
|   |-- external_data
|   |   |-- trainmap.csv
|   |   |-- traindata.csv
|   |   |-- testmap.csv
|   |   |-- testdata.csv
|   |-- trainmap.csv
|   |-- traindata.csv
|   |-- testmap.csv
|   |-- testdata.csv
|-- code
|   |-- main.py
|-- submit
|   |-- submit_20240915.csv
```

## Data Description
- **`data/` Folder:** Contains the original files provided for the competition.
- **`external_data/` Folder:** 
  - This folder can be used for external data sources. 
  - **Note:** We have not used additional samples from external datasets. Instead, we have calculated additional attributes based on the existing properties. For instance, we computed the matching degree between siRNA and target mRNA, which provides valuable insights for the prediction model. This approach allows us to enhance the dataset without increasing the number of samples.

## Usage Instructions
1. **Install Dependencies:**
   ```bash
   conda env create -f environment.yaml
   ```
   Activate the environment:
   ```bash
   conda activate clean_uniport
   ```

2. **Run the Code:**
   Execute the main script to start the prediction process:
   ```bash
   python code/main.py
   ```

   Ensure that the script is configured to read input files from the `data/` folder and save the results to the `submit/` folder.

## Results Output
- Results will be saved in the `submit/` folder with filenames in the format `submit_YYYYMMDD_HHMMSS.csv`.

## Random Seed
- To ensure reproducibility, random seeds are set in the code. If your results vary significantly, it will affect the evaluation.

## Notes
- Ensure all code is properly commented to facilitate code review.
- Use relative paths for file reading, such as `../data/XX`.

## Contact Information
For any questions or issues, please contact [Your Name](your.email@example.com).
```
