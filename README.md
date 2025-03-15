
# BDRP PROJECT
Project title: Enhancing Self-Supervised Learning for Image Clustering using Geometric Deep Learning

### Research Question
Can the integration of geometric deep learning techniques, specifically Group Equivariant Convolutional Networks,
improve deep clustering performance in medical imaging compared to conventional CNN-based approaches that rely on data augmentation?

## Overview
This repository integrates [Deep Clustering](https://arxiv.org/abs/1807.05520) and [Group Equivariant Convolutional Networks (G-CNNs)](https://arxiv.org/abs/1602.07576) for unsupervised visual feature learning. It utilizes the [NIH Chest X-Ray dataset](https://www.kaggle.com/datasets/nih-chest-xrays/data), consisting of 112,120 frontal-view X-ray images across 14 thoracic diseases.

## Methodology Pipeline

![Methodology Pipeline](results/methodology_pipeline.png)

## Repository Structure

The repository is organized into several high-level directories:

- **dataset/**  
  Contains the preprocessed files our experimental datasets.

- **experiments/**  
  Contains experiment scripts for running the deep clustering experiments.

- **evaluation/**  
  Provides scripts to assess clustering performance, compute evaluation metrics, and visualize the learned features.

- **slurm_scripts/**  
  Contains SLURM job scripts for running experiments on the GPU cluster. These scripts are pre-configured with the necessary commands to launch jobs in a ruche cluster environment.

- **requirements.txt**  
  Lists all the required Python packages and dependencies needed to run the project.

## Reproducing the Experiments

### Environment Setup on Ruche Cluster

To run the experiments on the ruche cluster, load the following modules and activate the appropriate conda environment:

```bash
module load cuda/11.7.0/gcc-11.2.0
module load python/3.9.10/gcc-11.2.0
module load anaconda3/2024.06/gcc-13.2.0
module load cudnn/8.6.0.163-11.8/oneapi-2023.2.1
source activate deepcluster
```
### SLURM Job Scripts

Training AlexNet Model
```bash
sbatch slurm_script/gpu_experiment1_alexnet.sh
```
Training P4M Model
```bash
sbatch slurm_script/gpu_experiment2_p4m_V1.sh
```

## Dependencies & Setup

### System Requirements

Ensure that your system meets the following specifications before installation:

| **Library**    | **Recommended Version**           |
| -------------- | --------------------------------- |
| CUDA Toolkit   | 11.7                              |
| cuBLAS         | 11.x (bundled with CUDA)          |
| PyTorch        | 1.13.1                            |
| Python         | 3.9.21                            |

### Required Dependencies

Install all required Python packages using the following command:

```bash
pip install -r requirements.txt
```

## Support, Feedback, Contributing
This project is open to feature requests, suggestions, and bug reports via [GitHub Issues](https://github.com/kamrulkonok/bdrp_project/issues). Contribution and feedback are encouraged and always welcome.  

For more information about how to contribute, the project structure, as well as additional contribution guidelines, see our [Contribution Guidelines](https://github.com/kamrulkonok/bdrp_project/blob/main/CONTRIBUTING.md).

## Acknowledgements
We would like to express our heartfelt gratitude to our supervisors [Akash Malhotra](https://www.linkedin.com/in/akash-malhotra13/) and Prof. [Nacéra Seghouani](https://www.linkedin.com/in/nac%C3%A9ra-seghouani-65454013/), for their invaluable guidance, continuous support, and insightful feedback throughout the project's development.
