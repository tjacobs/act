echo "TODO"
./imitate_episodes.py --task_name real_transfer_cube --ckpt_dir checkpoints --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 1 --lr 1e-5 --seed 0 --eval
