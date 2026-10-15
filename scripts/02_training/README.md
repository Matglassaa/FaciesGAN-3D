Here is how you can run the refactored script on your Linux cluster. Since it's a cluster, I recommend using nohup (to keep it running if your SSH session disconnects) or
  submitting it via your cluster's job scheduler (like SLURM with sbatch).

  Here are the 3 ways you can use the new hybrid configuration system:

  Method 1: Pure CLI (Command Line Arguments)
  This is similar to how you ran it before, but now requiring the --run_name parameter to organize your outputs cleanly.

    1 nohup python -u scripts/GAN/gan_training.py \
    2     --run_name "experiment_01_cli" \
    3     --data_file "/path/to/dataset.h5" \
    4     --output_dir "/path/to/outputs" \
    5     --num_gpus 2 \
    6     --epochs 50 \
    7     --batch_size 8 \
    8     --val_batch_size 8 \
    9     --validation_size 0.1 \
   10     --disable_one_hot \
   11     > training.log 2>&1 &

  Method 2: Pure JSON Configuration
  This is great for reproducibility. You create a JSON file with your parameters, and point the script to it.

  1. Create a file named my_config.json:

    1 {
    2     "run_name": "experiment_02_json",
    3     "data_file": "/path/to/dataset.h5",
    4     "output_dir": "/path/to/outputs",
    5     "epochs": 50,
    6     "batch_size": 8,
    7     "val_batch_size": 8,
    8     "num_gpus": 2,
    9     "validation_size": 0.1,
   10     "disable_one_hot": true
   11 }

  2. Run the script using the --config flag:

   1 nohup python -u scripts/GAN/gan_training.py --config my_config.json > training.log 2>&1 &

  Method 3: Hybrid (JSON + CLI Overrides)
  This is the most powerful method for HPC environments. You can have a "base" configuration file with all your standard defaults, and just override the specific parameters you
  want to change for a particular run via the CLI.

   1 nohup python -u scripts/GAN/gan_training.py \
   2     --config base_config.json \
   3     --run_name "experiment_03_hybrid" \
   4     --epochs 100 \
   5     --batch_size 16 \
   6     > training.log 2>&1 &
  (In this example, the script loads everything from base_config.json, but updates the run name, changes epochs to 100, and changes batch size to 16).

  Note for SLURM (Job Schedulers)
  If your Linux cluster uses SLURM to submit jobs instead of running them directly on the login node, you would simply place the python command (without nohup or &) inside your .sh
  batch script:

    1 #!/bin/bash
    2 #SBATCH --job-name=gan_training
    3 #SBATCH --output=training_%j.log
    4 #SBATCH --gpus=2
    5
    6 # Activate your environment
    7 source activate fluvgan
    8
    9 # Run the script
   10 python -u scripts/GAN/gan_training.py --config my_config.json