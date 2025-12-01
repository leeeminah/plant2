![Teaser](PlanT/PlanT2_teaser.png)
# PlanT 2.0: Exposing Biases and Structural Flaws in Closed-Loop Driving

<p align="center">
  <h3 align="center">
    <a href="https://arxiv.org/abs/2511.07292"> Paper</a> | <a href="https://huggingface.co/datasets/SimonGer/PlanT2_Dataset">Dataset</a> | <a href="https://huggingface.co/SimonGer/PlanT2">Model</a> 
  </h3>
</p>

# Contents
* [Installation / Setup](#Installation-/-Setup)
* [Quick start](#Quick-start)
* [Data generation](#data-generation)
* [Evaluation](#Evaluation)
* [Training](#Training)
* [Dataset](#Dataset)
* [Citation](#citation)

# Installation
```bash
# 1. Clone this repository
git clone https://github.com/autonomousvision/plant2.git
cd plant2

# 2. Setup Carla
# if you already have carla, skip this step
chmod +x setup_carla.sh
./setup_carla.sh

# 3. Setup environment
conda env create -f environment.yml
conda activate plant2
```

# Quick start inference
You can test the performance of PlanT 2.0's [pretrained models](https://huggingface.co/SimonGer/PlanT2) using the CARLA leaderboard evaluator. 
First, the following evironment variables have to be set correctly:
```bash
export CARLA_ROOT=/path/to/CARLA
export WORK_DIR=/path/to/plant2
export SCENARIO_RUNNER_ROOT=$WORK_DIR/scenario_runner_autopilot
export LEADERBOARD_ROOT=$WORK_DIR/leaderboard_autopilot
export PYTHONPATH=$CARLA_ROOT/PythonAPI/carla:$LEADERBOARD_ROOT:$SCENARIO_RUNNER_ROOT:$WORK_DIR/PlanT:$WORK_DIR/carla_garage
```
Afterwards, the evaluation can be started using two separate terminals:
```bash
# Terminal 1: Start CARLA, e.g. using:
$CARLA_ROOT/CarlaUE4.sh

# Terminal 2: Run the evaluation:
# Fill in your own paths and eval args
export PLANT_VIZ=/path/to/viz_[ROUTE]
export PLANT_CHECKPOINT=/path/to/checkpoint.ckpt
python leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py \
  --routes=[ROUTE_FILE] \
  --track=[SENSORS if B2D else MAP] \
  --agent=PlanT/PlanT_agent.py
```

# Evaluation
To perform a full evaluation of a benchmark on a SLURM cluster, you can refer to [`plant_evaluate.py`](PlanT/plant_evaluate.py). The script contains some config parameters that can be changed to evaluate different benchmarks, as well as some cluster-dependent settings.
After updating the parameters you can run the script, e.g. using:
```bash
python plant_evaluate.py \
  --checkpoint /path/to/epoch=029_final_1.ckpt \
  --routes /path/to/longest6_split/ \
  --out_root results/longest6 \
  --seeds 1 2 3
```

# Training 
To train PlanT 2.0 from scratch, you can use [`train_plant.sh`](PlanT/train_plant.sh), which can run locally or on a SLURM cluster. Relevant configuration files for training and model settings are [`config.yaml`](PlanT/config/config.yaml), [`PlanT.yaml`](PlanT/config/model/PlanT.yaml) and your user specific [`username.yaml`](PlanT/config/user/simon.yaml) (which is referenced in config.yaml). 

# Dataset
The dataset used during our experiments is availabe for download [on Huggingface](https://huggingface.co/datasets/SimonGer/PlanT2_Dataset). If you want to collect your own dataset, we provide the modified [`autopilot.py`](carla_garage/autopilot.py) and [`data_agent.py`](carla_garage/data_agent.py) used for our dataset. You can use [`collect_dataset_slurm.py`](0_run_collect_dataset_slurm.sh) or [`0_run_collect_dataset_slurm.sh`](0_run_collect_dataset_slurm.sh) start the dataset collection process on a SLURM cluster. The scripts have to be modified according to your cluster setup. The current settings do not require any GPU's and instead run small jobs with 2 CPU's and 20Gb of RAM, since they don't collect RGB information.

# Citation
```latex
@misc{gerstenecker2025plant20exposingbiases,
      title={PlanT 2.0: Exposing Biases and Structural Flaws in Closed-Loop Driving}, 
      author={Simon Gerstenecker and Andreas Geiger and Katrin Renz},
      year={2025},
      eprint={2511.07292},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2511.07292}, 
}
```

# Acknowlegdements
This repository builds upon the work of [`carla_garage`](https://github.com/autonomousvision/carla_garage) and [`PlanT`](https://github.com/autonomousvision/plant). We thank the authors for open-sourcing their work.
