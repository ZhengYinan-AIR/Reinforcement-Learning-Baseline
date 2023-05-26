import argparse
from utils.util import boolean

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', default='PPO', type=str)
    parser.add_argument('--environment', default='safety-gym', type=str)
    parser.add_argument('--robot', default='Point', type=str)
    parser.add_argument('--task', default='Goal1', type=str)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--wandb', default=True, type=boolean)
    parser.add_argument('--seed', default=10, type=int)

    parser.add_argument('--generate_datasets', default=False, type=boolean)
    parser.add_argument('--num_unsafe_obs', default=500, type=int)
    parser.add_argument('--num_data_episodes', default=1000, type=int)
    parser.add_argument('--data_type', default='expert', type=str)

    parser.add_argument('--use_reward_scale', default=True, type=boolean)
    parser.add_argument('--reward_scale', default=50.0, type=float)
    parser.add_argument('--use_constraint_scale', default=True, type=boolean)
    parser.add_argument('--constraint_scale', default=10.0, type=float)
    parser.add_argument('--constraint_offset', default=0.5, type=float)


    parser.add_argument('--total_iterations', default=int(1e6), type=int)
    parser.add_argument('--log_iterations', default=int(5e3), type=int)
    parser.add_argument('--pre_train_iterations', default=int(-1), type=int)
    parser.add_argument('--policy_freq', default=2, type=int)

    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--alpha', default=5.0, type=float)
    parser.add_argument('--tau', default=0.005, type=float)
    parser.add_argument('--epsilon', default=0.1, type=float)
    parser.add_argument('--hidden_sizes', default=256)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--f', default='kl', type=str)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--actor_lr', default=3e-5, type=float)
    parser.add_argument('--penalty_lr', default=5e-3, type=float)
    parser.add_argument('--v_l2_reg', default=0.0001, type=float)
    parser.add_argument('--grad_norm_clip', default=0.5, type=float)
    parser.add_argument('--use_policy_entropy_constraint', default=True, type=boolean)
    parser.add_argument('--target_entropy', default=None, type=float)
    parser.add_argument('--use_behavior_policy', default=False, type=boolean)

    return parser