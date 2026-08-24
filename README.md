# GRSDNet
Learnable Guided Residual Spectral Decomposition for Industrial Anomalous Sound Detection

## Citation
If you use this code in your research, please cite our paper:

## Overview
This paper proposes GRSDNet, a learnable guided residual spectral decomposition method for industrial anomalous sound detection:
- A guided residual spectral decomposition method improves anomalous sound detection.
- Local gated patch attention strengthens anomaly-relevant spectral representations.
- GRSDNet improves cross-domain robustness with marginal computational overhead.

## Project Structure

├── .gitattributes        # Git configuration for file attributes and repository management

├── config.py             # Configuration utilities for loading and managing experiment parameters

├── config.yaml           # Global experiment configuration, including model and training settings

├── model.py              # Implementation of the proposed model and its main components

└── README.md             # Project overview, environment setup, usage, and reproduction instructions


## Requirements
We use Conda python 3.8+ and strongly recommend that you create a new environment.
* Prerequisite: Python 3.8 or higher versions
```shell script
conda create -n GRSDNet python=3.8
conda activate GRSDNet
```

## Environment
This code is tested using Python 3.8, Pytorch 1.10, and CUDA 11.1
* Install all packages in the requirement.txt
```shell script
pip3 install -r requirements.txt
```

## Datasets

### DCASE 2022 
More details can be find in this [link](https://dcase.community/challenge2022/index). please request and download the data from the original WORKSHOP.

### DCASE 2024 
More details can be find in this [link](https://dcase.community/challenge2024/index). please request and download the data from the original WORKSHOP.

### DCASE 2024 
More details can be find in this [link](https://dcase.community/challenge2025/index). please request and download the data from the original WORKSHOP.

## Quick Start
1. Configure Datasets
Place your hyperspectral datasets in .mat format in the datasets/ directory

2. Update config.py to add your dataset paths:

3. Run the Full Pipeline:
python model.py

4. Output Results
All results are automatically saved in the ./results/{dataset_name}/ directory:


## Get Involved
Should you have any query please contact me.
Please create a GitHub issue if you have any questions, suggestions, requests or bug-reports. 
Don't hesitate to send us an e-mail or report an issue, if something is broken or if you have further questions.



