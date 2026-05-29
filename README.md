# Moo-cross
A Multi-Objective Optimization Model for Cross-dataset Network Intrusion Detection

## Authors
- Leonardo de Freitas Rocha
- Humberto Fontoura Pradera
- Eduardo Kugler Viegas
- Altair Olivo Santin

## Abstract
<p style="text-align: justify;">
    Over the past years, several highly accurate Machine Learning (ML) schemes for Network Intrusion Detection Systems (NIDS) have been proposed.
However, despite achieving promising results within their training domains, these models often fail to generalize across different network environments due to dataset heterogeneity and varying feature distributions.
To address this issue, we propose a two-phase multi-objective optimization framework designed to improve cross-dataset generalization.
First, cross-domain performance is formulated as a multi-objective optimization problem that maximizes accuracy across multiple domains concurrently.
Second, model training is performed through joint hyperparameter tuning and feature selection, enabling the discovery of configurations that generalize well across datasets.
Using a genetic search-based approach, our method evaluates each candidate solution across diverse domains to identify stable operating points that balance accuracy and generalization.
Experimental results on three benchmark NetFlow datasets show that our approach improves cross-dataset F1 Scores by up to 0.74 compared to traditional single-domain training.
Moreover, when compared to models trained on all datasets simultaneously, our framework increases the average F1-Score by 0.15, demonstrating enhanced generalization across domains.</p> 

## Proposal
<img width="1431" height="622" alt="image" src="https://github.com/user-attachments/assets/700f0910-17af-4471-bcce-c1ce4395ef39" />

## Datasets
NetFlow v3: NF-ToN-IoT-v3, NF-CSE-CIC-IDS-v3 and NF-BoT-IoT-v3  
They were published at https://staff.itee.uq.edu.au/marius/NIDS_datasets/

## Files and Directory structure
Files over 25 Mb are zipped.  
Directory structure:
- **linux**: files used for running the experiment on linux environment
- **resultados**: output from experiment and graphs
- **apoio**: Jupyter Notebooks used to process output files creating graphs and bringing insights 
  
Main file to run the experiment: \linux\python\experimentos_v12.py

## Citation
Article proposed at IEEE Globecom 2026
