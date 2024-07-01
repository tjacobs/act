# ACT: Action Chunking with Transformers

### Overview
This contains the implementation of ACT, Action Chunking with Transformers.
For training robot arms to perform tasks.

It has two simulated environments: Transfer Cube and Bimanual Insertion.

You can train and evaluate ACT in sim or real.

### Install

On an Apple Silicon Mac:

    # Install brew
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

    # Install anaconda, https://www.anaconda.com/download/success
    curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
    sh Miniconda3-latest-MacOSX-arm64.sh

    # Install PyTorch for metal
    conda install pytorch torchvision torchaudio -c pytorch-nightly

    # Clone ACT from tjacobs
    cd ~/Documents/GitHub/
    git clone https://github.com/tjacobs/act.git
    cd act

    # Create python 3.8.10 env
    conda config --append channels conda-forge
    conda create -n aloha python=3.8.10
    conda activate aloha
    pip install torchvision
    pip install torch
    pip install pyquaternion
    pip install pyyaml
    pip install rospkg
    pip install pexpect
    pip install mujoco==2.3.7
    pip install dm_control==1.0.14
    pip install opencv-python
    pip install matplotlib
    pip install einops
    pip install packaging
    pip install h5py
    pip install ipython
    cd detr && pip install -e .
    cd ..

### Run

    # Generate training runs
    python3 record_sim_episodes.py --task_name sim_transfer_cube_scripted --dataset_dir data/sim_transfer_cube_scripted --num_episodes 2 --onscreen_render 

    # Train network
    python3 imitate_episodes.py    --task_name sim_transfer_cube_scripted --ckpt_dir checkpoints --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 2000 --lr 1e-5 --seed 0

    # Evaluate network
    python3 imitate_episodes.py    --task_name sim_transfer_cube_scripted --ckpt_dir checkpoints --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 2000 --lr 1e-5 --seed 0 --eval

### Tips

Other task: sim_insertion_scripted

To vizualize:
    python3 visualize_episodes.py --dataset_dir <data save dir> --episode_idx 0

The success rate should be around 90% for transfer cube, and around 50% for insertion.

To enable temporal ensembling, add flag ``--temporal_agg``.

Videos will be saved to ``checkpoints`` for each rollout.

You can also add ``--onscreen_render`` to see real-time rendering during evaluation.

For real-world, train for at least 5000 epochs or 3-4 times the length after the loss has plateaued.

Refer to [tuning tips](https://docs.google.com/document/d/1FVIZfoALXg_ZkYKaYVh-qOlaXveq5CtvJHXkY25eYhs/edit?usp=sharing).

For real, you would also need to install [ALOHA](https://github.com/tonyzhaozh/aloha).

You can find all scripted/human demo for simulated environments [here](https://drive.google.com/drive/folders/1gPR03v05S1xiInoVJn7G7VJ9pDCnxq9O?usp=share_link).

### Repo Structure
- ``record_sim_episodes.py`` Record scripted episodes from sim
- ``imitate_episodes.py`` Train and Evaluate ACT
- ``visualize_episodes.py`` Save videos from a .hdf5 dataset
- ``policy.py`` An adaptor for ACT policy
- ``detr`` Model definitions of ACT, modified from DETR
- ``sim_env.py`` Mujoco + DM_Control environments with joint space control
- ``ee_sim_env.py`` Mujoco + DM_Control environments with EE space control
- ``scripted_policy.py`` Scripted policies for sim environments
- ``constants.py`` Constants shared across files
- ``utils.py`` Utils such as data loading and helper functions


