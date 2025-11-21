NUM_PROCS=2

torchrun --nproc_per_node=${NUM_PROCS} scripts/baseline/opal/train_opal.py 