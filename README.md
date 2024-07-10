# ACT: Action Chunking with Transformers

### Overview
This contains the implementation of ACT, Action Chunking with Transformers.
For training robots to perform tasks.

It has two simulated environments: Transfer Cube and Insertion.

You can train and evaluate ACT in sim or real.

### Install

On an Apple Silicon Mac:

    # Install brew
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

    # Install anaconda, https://www.anaconda.com/download/success
    curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
    sh Miniconda3-latest-MacOSX-arm64.sh

    # Install PyTorch for Apple Mac metal
    conda install pytorch torchvision torchaudio -c pytorch-nightly

On x64 linux:

    # Install conda
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm -rf ~/miniconda3/miniconda.sh
    #echo "PATH=$PATH:~/miniconda3/bin" >> ~/.bashrc
    ~/miniconda3/bin/conda init

Then:

    # Clone ACT from tjacobs
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

    # Generate training data from simulator
    ./record_sim.sh

    # Train network from data (sim or real)
    ./train.sh

    # Evaluate network in simulator
    ./evaluate_sim.sh

    # Generate training data from real robot
    ./record_real.sh

    # Evaluate network on real robot
    ./evaluate_real.sh

### Tips

The evaluation success rate should be around 90% for transfer cube simulator, and around 50% for insertion simulator.

To enable temporal ensembling, add flag ``--temporal_agg``.

Videos will be saved to ``checkpoints`` for each evaluation.

For real, train for at least 5000 epochs or 3-4 times the length after the loss has plateaued.

Refer to [tuning tips](https://docs.google.com/document/d/1FVIZfoALXg_ZkYKaYVh-qOlaXveq5CtvJHXkY25eYhs/edit?usp=sharing).

For real, you would also need to install [ALOHA](https://github.com/tonyzhaozh/aloha).

You can find recorded data for scripted/human sim [here](https://drive.google.com/drive/folders/1gPR03v05S1xiInoVJn7G7VJ9pDCnxq9O?usp=share_link).

### Repo Structure
- ``record_sim_episodes.py`` Record data from sim
- ``imitate_episodes.py`` Train and evaluate ACT
- ``policy.py`` An adaptor for ACT policy
- ``detr`` Model definitions of ACT, modified from DETR
- ``sim_env.py``    Mujoco + DM_Control environments with joint space control
- ``ee_sim_env.py`` Mujoco + DM_Control environments with end effector space control
- ``scripted_policy.py`` Scripted policies for sim environments
- ``constants.py`` Constants shared across files
- ``utils.py`` Utils such as data loading and helper functions
- ``visualize_episodes.py`` Save videos from a .hdf5 dataset

