# Harnessing Dataset Cartography for Better Compositional Generalization in Transformers

The official repository for our EMNLP 2023 Findings paper [Harnessing Dataset Cartography for Better Compositional Generalization in Transformers](ospanbatyr.github.io). This repository is based on the [GitHub repo](https://github.com/RobertCsordas/transformer_generalization) of the EMNLP 2021 paper [The Devil is in the Detail: Simple Tricks Improve Systematic Generalization of Transformers](https://aclanthology.org/2021.emnlp-main.49.pdf) by Csordás et al. We are grateful for their clean, modular, and easily modifiable codebase.

Kindly be aware that this repository is a refined edition of our internal research repository. Should you encounter any issues, feel free to reach out to me without hesitation.

## Setup

This project requires Python 3 (tested with Python 3.6) and PyTorch 1.9.

```bash
conda env create -f environment.yml
conda activate cartography4comp
pip install -r requirements.txt
```

Create a Weights and Biases account and run 
```bash
wandb login
```

More information on setting up Weights and Biases can be found on
https://docs.wandb.com/quickstart.

### Downloading data

All datasets are downloaded automatically except CFQ which is hosted in Google Cloud and one has to log in with their Google account to be able to access it.

#### CFQ
Download the .tar.gz file manually from here:

https://storage.cloud.google.com/cfq_dataset/cfq1.1.tar.gz

Copy it to the ``cache/CFQ/`` folder. You should have a ``cache/CFQ/cfq1.1.tar.gz`` file in the project folder if you did everyhing correctly. 


## Usage

### Running the experiments from the paper on a cluster

The code makes use of Weights and Biases for experiment tracking. In the ```sweeps``` directory, we provide sweep configurations for all experiments we have performed. The sweeps are officially meant for hyperparameter optimization, but we use them to run full trainings and .

To reproduce our results, start a sweep for each of the YAML files in the ```sweeps``` directory. Run wandb agent for each of them in the _root directory of the project_. This will run all the experiments, and they will be displayed on the W&B dashboard. The name of the sweeps must match the name of the files in ```sweeps``` directory, except the ```.yaml``` ending. More details on how to run W&B sweeps can be found at https://docs.wandb.com/sweeps/quickstart.

For example, if you want to run CFQ full training experiments and save the training dynamics , run ```wandb sweep --name cfq_mcd sweeps/cfq_mcd.yaml```. This creates the sweep and prints out its ID. Then run ```wandb agent <ID>``` with that ID. To use dataset cartography and create new subsets, you first need to run the full training as dataset cartography uses the training dynamics of the full training. 

#### Running subset experiments

If you want to bypass this step and directly want to reproduce our results, you can run subset experiments with our previously created subsets. For example, to run the experiments for the CFQ CHIA hard-to-learn 33% sized subset, you can directly run the ```wandb sweep --name cfq_mcd_subset sweeps/chia/htl/cfq_mcd_subset.yaml``` command. 50% and subset combinations can be found at ```wandb sweep --name cfq_mcd_subset sweeps/chia/0_5/htl/cfq_mcd_subset.yaml``` and  ```wandb sweep --name cfq_mcd_subset sweeps/chia/comb/htl_amb/cfq_mcd_subset.yaml``` respectively. If you want to run experiments with different seeds and subsets, do not forget to change the seed and the subset address.

#### Running curriculum  experiments

For curriculum learning (CL) experiments, we created different branches. For the CL framework in Hacohen and Weinshal (2019), please checkout to the **curr_exp** branch. For the Zhang et al. (2020) CL framework, please checkout to the **curr_lin** branch. The curriculum configs are under `sweeps/currclm` directory.  For example, to run the experiments for the CFQ CHIA hard-to-learn curriculum, you can directly run the ```wandb sweep --name cfq_mcd_subset sweeps/currclm/chia/htl/cfq_mcd_subset.yaml``` command. If you want to run experiments with seeds and subsets different than in the configs, do not forget to change the seed and the subset address.

#### Creating subsets and plots from the paper
In order to create the subsets and plots from the paper, you must run the full trainings first to store training dynamics. Then, you can create 33%, 50%, and combination subsets in the paper by running `cartography/CreateSubsets.ipynb` notebook from start to end. Similarly, you can create cartography plots by running `cartography/Plots.ipynb` notebook. The newly created subsets and plots are created in the `subsets` and `plots` folders, respectively. 

To create curriculums, you can use the  `cartography/CreateCurriculums.ipynb` notebook. Creating curriculum plots take a bit more time. First, you need to extract the running test accuracy from WandB for the runs that you want in the CSV format. After putting these CSV files under `cartography/plots/curriculum_plots` in the described format and execute `CurriculumPlots.ipynb` notebook completely, you will create the curriculum plots. Please check `cartography/plots_paper` and `cartography/subsets_paper` for more information.

### Reducing memory usage

In case some tasks won't fit on your GPU, play around with "-max_length_per_batch <number>" argument. It can trade off memory usage/speed by slicing batches and executing them in multiple passes. Reduce it until the model fits.
  
### BibTex
```
@misc{i̇nce2023harnessing,
      title={Harnessing Dataset Cartography for Improved Compositional Generalization in Transformers}, 
      author={Osman Batur İnce and Tanin Zeraati and Semih Yagcioglu and Yadollah Yaghoobzadeh and Erkut Erdem and Aykut Erdem},
      year={2023},
      eprint={2310.12118},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}```
