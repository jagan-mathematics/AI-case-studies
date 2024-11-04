#!/bin/bash
set -e

# Start timer
start_time=$(date +%s)

# Create environment name with the current date
env_prefix=ai_experimentation_env

# Create the conda environment

if [[ " $@ " =~ " --only-packages " ]]; then
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install ninja
    pip install git+https://github.com/facebookresearch/xformers.git@d3948b5cb9a3711032a0ef0e036e809c7b08c1e0#egg=xformers
    pip install --requirement requirements.txt
    exit 0;
fi

source $CONDA_ROOT/etc/profile.d/conda.sh

if conda info --envs | grep -q "$env_prefix"; then
    echo "Environment '$env_prefix' already exists. Activating..."
else
    echo "Creating environment '$env_prefix'..."
    conda create -n $env_prefix python=3.11 -y -c anaconda
    # Install packages
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install ninja
    pip install git+https://github.com/facebookresearch/xformers.git@d3948b5cb9a3711032a0ef0e036e809c7b08c1e0#egg=xformers
    pip install --requirement requirements.txt
fi

conda activate $env_prefix

echo "Currently in env $(which python)"

# End timer
end_time=$(date +%s)

# Calculate elapsed time in seconds
elapsed_time=$((end_time - start_time))

# Convert elapsed time to minutes
elapsed_minutes=$((elapsed_time / 60))

echo "Environment $env_prefix created and all packages installed successfully in $elapsed_minutes minutes!"
