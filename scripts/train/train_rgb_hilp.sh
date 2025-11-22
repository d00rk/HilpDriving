NUM_PROCS=2
DEVICES=2,3

CUDA_VISIBLE_DEVICES=${DEVICES} HYDRA_FULL_ERROR=1 nohup torchrun --nproc_per_node=${NUM_RPOCS} scripts/train/train_rgb_hilp.py > nohup.out 2>&1 &