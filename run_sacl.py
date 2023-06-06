import os
import gym
import safety_gym
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import utils
from utils import boolean
from env.sg.sg import SafetyGymWrapper
from run_sac import SAC, ReplayBuffer

import wandb
from tqdm import tqdm

from network import Double_Critic, Scalar_Multiplier, MLP_Multiplier
    
class SACL(SAC):
    def __init__(self, state_dim, action_dim, max_action, config):
        super().__init__(state_dim, action_dim, max_action, config)

        self.multiplier_update_interval = config['multiplier_update_interval']
        self.multiplier_lr = config['multiplier_lr']
        self.multiplier_lr_end = config['multiplier_lr_end']
        self.cost_limit = config['cost_limit']
        self.multiplier_ub = config['multiplier_ub'] # lambda upper bound
        self.penalty_ub = config['penalty_ub']
        self.penalty_lb = config['penalty_lb']

        self.cost_critic = Double_Critic(state_dim, action_dim, self.hidden_width, use_softplus=config['use_softplus']).to(self.device)
        self.cost_critic_target = copy.deepcopy(self.cost_critic).to(self.device)

        self.cost_critic_optimizer = torch.optim.Adam(self.cost_critic.parameters(), lr=self.lr)
        self.cost_critic_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.cost_critic_optimizer,
            T_max=self.updates_per_training,
            eta_min=self.lr_end
        )

        # multiplier
        if config['use_mlp_multiplier']:
            self.multiplier_updates_num = int(self.updates_per_training / self.multiplier_update_interval)
            self.multiplier = MLP_Multiplier(state_dim, self.hidden_width, config['multiplier_ub']).to(self.device)
            self.multiplier_optimizer = torch.optim.Adam(self.multiplier.parameters(), lr=self.multiplier_lr)
            self.multiplier_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.multiplier_optimizer,
                T_max=self.multiplier_updates_num,
                eta_min=self.multiplier_lr_end
            )
        else:
            self.multiplier_updates_num = int(self.updates_per_training / self.multiplier_update_interval)
            self.multiplier = Scalar_Multiplier(init_value=config['multiplier_init']).to(self.device)
            self.multiplier_optimizer = torch.optim.Adam(self.multiplier.parameters(), lr=self.multiplier_lr)
            self.multiplier_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.multiplier_optimizer,
                T_max=self.multiplier_updates_num,
                eta_min=self.multiplier_lr_end
            )

    def update_cost_critic(self, batch_s, batch_a, batch_s_, batch_dw, batch_c, result): # not need batch_r！

        with torch.no_grad():
            batch_a_, log_pi_ = self.actor(batch_s_)  # a' from the current policy
            # Compute target Qc
            target_Qc1, target_Qc2 = self.cost_critic_target(batch_s_, batch_a_)
            target_Qc = batch_c + self.GAMMA * (1 - batch_dw) * torch.max(target_Qc1, target_Qc2) # qc use the torch.max

        # Compute current Qc
        current_Qc1, current_Qc2 = self.cost_critic(batch_s, batch_a)
        # Compute cost_critic loss
        cost_critic_loss = F.mse_loss(current_Qc1, target_Qc) + F.mse_loss(current_Qc2, target_Qc)

        result.update({
            'cost_critic_loss': cost_critic_loss,
            'current_qc': current_Qc1.detach().mean(),
            'target_qc': target_Qc.detach().mean(),
            'c': batch_c.detach().mean(),
        })
        return result
    
    def update_actor(self, batch_s, result):

        # Compute actor loss
        a, log_pi = self.actor(batch_s)
        Q1, Q2 = self.critic(batch_s, a)
        Q = torch.min(Q1, Q2)
        uncstr_actor_loss = (self.alpha * log_pi - Q).mean()

        # Compute alpha loss
        if self.adaptive_alpha:
            # We learn log_alpha instead of alpha to ensure that alpha=exp(log_alpha)>0
            alpha_loss = -(self.log_alpha().exp() * (log_pi + self.target_entropy).detach()).mean()

        # Constrained part
        if self.cost_critic.critic_num() == 2:
            Qc1, Qc2 = self.cost_critic(batch_s, a)
            Qc= torch.max(Qc1, Qc2)
        else:
            Qc = self.cost_critic(batch_s, a)
        violation = Qc - self.cost_limit
        violation = torch.clip(violation, min=self.penalty_lb, max=self.penalty_ub)
        if self.multiplier.func_type() == 'scalar':
            multiplier = torch.clip(self.multiplier(), 0, self.multiplier_ub)
        elif self.multiplier.func_type() == 'mlp':
            multiplier = self.multiplier(batch_s)

        cstr_actor_loss= torch.mean(multiplier.detach() * violation)

        actor_loss = uncstr_actor_loss + cstr_actor_loss
        # actor_loss = uncstr_actor_loss


        result.update({
            'uncstr_actor_loss': uncstr_actor_loss,
            'cstr_actor_loss': cstr_actor_loss,
            'actor_loss': actor_loss,
            'alpha_loss': alpha_loss,
            'alpha': self.log_alpha().exp().detach().mean(),
            'actor_qc': Qc.detach().mean(),
        })
        return result
    
    def update_multiplier(self, batch_s, result):
        # Constrained part
        a, _ = self.actor(batch_s)
        if self.cost_critic.critic_num() == 2:
            Qc1, Qc2 = self.cost_critic(batch_s, a)
            Qc= torch.max(Qc1, Qc2)
        else:
            Qc = self.cost_critic(batch_s, a)
        violation = Qc - self.cost_limit
        violation = torch.clip(violation, min=self.penalty_lb, max=self.penalty_ub)

        if self.multiplier.func_type() == 'scalar':
            multiplier = torch.clip(self.multiplier(), 0, self.multiplier_ub)
            multiplier_loss = -torch.mean(multiplier * violation.detach()) 

            result.update({
                'multiplier_loss': multiplier_loss,
                'multiplier': multiplier.detach().mean(),
                'violation': violation.detach().mean(),
            })

        elif self.multiplier.func_type() == 'mlp':
            multiplier = self.multiplier(batch_s)

            multiplier_safe = torch.mul(violation<=0, multiplier)
            multiplier_unsafe = torch.mul(violation>0, multiplier)

            unsafe_target = ((violation>0) * self.multiplier_ub).type(torch.float32)
            unsafe_lam_loss = F.mse_loss(multiplier_unsafe, unsafe_target)
            safe_lam_loss = -torch.mean(torch.mul(multiplier_safe, violation.detach()))
            multiplier_loss = safe_lam_loss + unsafe_lam_loss

            result.update({
                'multiplier_loss': multiplier_loss,
                'unsafe_lam_loss': unsafe_lam_loss,
                'safe_lam_loss': safe_lam_loss,
                'safe_l': multiplier[violation[...,0]<=0].mean(),
                'unsafe_l': multiplier[violation[...,0]>0].mean(),
                'violation_unsafe': violation[violation[...,0]<=0].mean(),
                'violation_safe': violation[violation[...,0]>0].mean()
            })


        return result
    
    def learn(self, batch_s, batch_a, batch_r, batch_s_, batch_dw, batch_c, batch_cv, result={}):
        # Compute critic loss
        result = self.update_critic(batch_s, batch_a, batch_r, batch_s_, batch_dw, result)
        result = self.update_cost_critic(batch_s, batch_a, batch_s_, batch_dw, batch_c, result)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        result['critic_loss'].backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm)
        self.critic_optimizer.step()
        self.critic_lr_scheduler.step()

        self.cost_critic_optimizer.zero_grad()
        result['cost_critic_loss'].backward()
        torch.nn.utils.clip_grad_norm_(self.cost_critic.parameters(), max_norm=self.grad_norm)
        self.cost_critic_optimizer.step()
        self.cost_critic_lr_scheduler.step()

        if self.num_update % self.actor_update_interval == 0:

            result = self.update_actor(batch_s, result)

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            result['actor_loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm)
            self.actor_optimizer.step()
            self.actor_lr_scheduler.step()

            # Optimize the alpha
            if self.adaptive_alpha:
                self.alpha_optimizer.zero_grad()
                result['alpha_loss'].backward()
                torch.nn.utils.clip_grad_norm_(self.log_alpha.parameters(), max_norm=self.grad_norm)
                self.alpha_optimizer.step()
                self.alpha = self.log_alpha().exp()



        if self.num_update % self.multiplier_update_interval == 0:
            result = self.update_multiplier(batch_s, result)
            # Optimize the multiplier
            self.multiplier_optimizer.zero_grad()
            result['multiplier_loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.multiplier.parameters(), max_norm=self.grad_norm)
            self.multiplier_optimizer.step()
            self.multiplier_lr_scheduler.step()
        
        # Softly update target networks
        self.soft_update(self.critic, self.critic_target)
        self.soft_update(self.cost_critic, self.cost_critic_target)

        self.num_update += 1

        return result
    

def evaluate_policy(env, agent, env_name):
    times = 10  # Perform evaluations and calculate the average
    evaluate_reward = 0
    evaluate_cost = 0
    for _ in tqdm(range(0, times), ncols=70, desc='Evaluation', initial=1, total=times, ascii=True, disable=os.environ.get("DISABLE_TQDM", False)):
        s = env.reset()
        done = False
        episode_reward = 0
        episode_cost = 0
        while not done:
            a = agent.choose_action(s, deterministic=True)  # We use the deterministic policy during the evaluating
            s_, r, done, info = env.step(a)
            if env_name == 'simple_sg':
                c = info['violation']
            elif env_name == 'safety_gym':
                c = info['cost']
        
            episode_reward += r
            episode_cost += c

            s = s_
        evaluate_reward += episode_reward
        evaluate_cost += episode_cost

    return int(evaluate_reward / times), int(evaluate_cost / times)

def run(config):
    if config['environment'] == 'simple_sg':
        env_list = ['point', 'car']
        env_name = config['env_name']
        assert env_name in env_list, "Invalid Env"
        if config['violation_done']:
            env = SafetyGymWrapper(
                robot_type=env_name,
                id=None,    # train: done on violation; eval: false
            )
        else:
            env = SafetyGymWrapper(
                robot_type=env_name,
                id=1,    # train: done on violation; eval: false
            )

        env_evaluate = SafetyGymWrapper(
            robot_type=env_name,
            id=1,    # train: done on violation; eval: false
        )
        exp_name = config['algo'] + '_' + env_name

    elif config['environment'] == 'safety_gym':
        # Verify experiment
        robot_list = ['Point', 'Car', 'Doggo']
        task_list = ['Goal1', 'Goal2', 'Button1', 'Button2', 'Push1', 'Push2']
        algo = config['algo']
        task = config['task']
        robot = config['robot']
        assert task in task_list, "Invalid task"
        assert robot in robot_list, "Invalid robot"
        exp_name = algo + '_' + robot + task

        env_name = 'Safexp-' + robot + task + '-v0'
    
        env = gym.make(env_name)
        env_evaluate = gym.make(env_name)

    seed = config['seed']
    # Set random seed
    env.seed(seed)
    env.action_space.seed(seed)
    env_evaluate.seed(seed)
    env_evaluate.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    if config['environment'] == 'safety_gym':
        max_episode_steps = 1000
    else:
        max_episode_steps = env._max_episode_steps  # Maximum number of steps per episode

    train_steps = config['max_train_steps'] - config['random_steps'] # the total train steps

    config.update({
        'env_name': env_name,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'max_action': max_action,
        'max_episode_steps': max_episode_steps,
        'train_steps': train_steps,
    })

    
    if config['wandb'] == True:
        wandb.init(project=exp_name, name='test_s'+str(config['seed']), reinit=True, mode='online')

    # Prepare Logger
    logger_kwargs = utils.setup_logger_kwargs(exp_name, config['seed'])

    logger = utils.Logger(**logger_kwargs)
    logger.save_config(config)

    agent = SACL(state_dim, action_dim, max_action, config)
    replay_buffer = ReplayBuffer(state_dim, action_dim, config['device'])

    evaluate_num = 0  # Record the number of evaluations
    total_steps = 0  # Record the total steps during the training
    red_list = ['ep_r', 'ep_c']
    result = {}

    while total_steps < config['max_train_steps']:
        s = env.reset()
        episode_steps = 0
        done = False
        while not done:
            episode_steps += 1
            if total_steps < config['random_steps']:  # Take the random actions in the beginning for the better exploration
                a = env.action_space.sample()
            else:
                a = agent.choose_action(s)
            s_, r, done, info = env.step(a)

            if config['use_reward_scale']:
                r = r * config['reward_scale']

            if config['goal_met_done']:
                info_keys = info.keys()
                goal_met = ('goal_met' in info_keys)
                done = done or goal_met # collision not done, reach goal done and terminal done
            
            if config['environment'] == 'safety_gym':
                c = info['cost']
                cv = c
            else:
                c = info['violation']
                cv = torch.tensor(info['constraint_value'], dtype=torch.float)

            replay_buffer.store(s, a, r, s_, done, c, cv)  # Store the transition
            s = s_

            if total_steps >= config['random_steps']:
                result = agent.learn(*replay_buffer.sample(config['batch_size']))

            # Evaluate the policy every 'evaluate_freq' steps
            if total_steps % config['evaluate_freq'] == 0:

                logger.log('\n==========Start evaluation==========')
                evaluate_num += 1
                evaluate_reward, evaluate_cost = evaluate_policy(env_evaluate, agent, config['environment'])
                result.update({'ep_r': evaluate_reward, 'ep_c': evaluate_cost})
                for k, v in result.items():
                    if k in red_list:
                        logger.log(f'- {k:20s}:{v:5.5f}', color='red')
                    else:
                        print(f'- {k:20s}:{v:5.5f}')
                
                if config['wandb']:
                    wandb.log(result)
                
                result.update({'iteration': total_steps})
                # Save results
                logger.save_result_logs(result)

                # Save model
                dir = logger.get_dir()
                file_name = agent.save(dir)
                logger.log('Logging model to ' + file_name)
                logger.log('\n==========Start training==========')


            total_steps += 1

        logger.log('Complete one episode with {} steps'.format(total_steps), color='magenta')

    # Save model
    dir = logger.get_dir()
    file_name = agent.save(dir)
    logger.log('Logging model to ' + dir)

def get_parser():
    parser = argparse.ArgumentParser()
    # SAC config
    parser.add_argument('--algo', default='SACL', type=str)
    parser.add_argument('--environment', default='safety_gym', type=str) # simple_sg or safety_gym
    # parser.add_argument('--environment', default='simple_sg', type=str)
    parser.add_argument('--robot', default='Point', type=str)
    parser.add_argument('--task', default='Goal1', type=str)
    parser.add_argument('--env_name', default='point', type=str)
    parser.add_argument('--device', default='cuda:3', type=str)
    parser.add_argument('--wandb', default=True, type=boolean)
    parser.add_argument('--seed', default=1283, type=int)

    parser.add_argument('--max_train_steps', default=int(3e6), type=int)
    parser.add_argument('--evaluate_freq', default=int(5e3), type=int)
    parser.add_argument('--random_steps', default=int(5e3), type=int)

    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--tau', default=0.005, type=float)
    parser.add_argument('--adaptive_alpha', default=True, type=boolean)
    parser.add_argument('--hidden_sizes', default=256)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--lr', default=3e-4, type=float)
    parser.add_argument('--lr_end', default=8e-5, type=float)
    parser.add_argument('--actor_lr', default=8e-5, type=float)
    parser.add_argument('--actor_lr_end', default=4e-5, type=float)
    parser.add_argument('--actor_update_interval', default=int(1), type=int)
    parser.add_argument('--grad_norm', default=5., type=float)
    parser.add_argument('--alpha_init', default=0., type=float)
    parser.add_argument('--alpha_lr', default=8e-5, type=float)
    parser.add_argument('--goal_met_done', default=False, type=boolean)
    parser.add_argument('--violation_done', default=False, type=boolean)

    parser.add_argument('--use_reward_scale', default=True, type=boolean)
    parser.add_argument('--reward_scale', default=200., type=float)

    # multiplier
    parser.add_argument('--use_multiplier', default=False, type=boolean) # test the sac 
    parser.add_argument('--multiplier_lr', default=3e-4, type=float)
    parser.add_argument('--multiplier_lr_end', default=1e-5, type=float)
    parser.add_argument('--multiplier_update_interval', default=int(2), type=int)
    parser.add_argument('--multiplier_init', default=0.5, type=float)
    parser.add_argument('--multiplier_ub', default=25., type=float)
    parser.add_argument('--penalty_ub', default=100., type=float)
    parser.add_argument('--penalty_lb', default=-1., type=float)
    parser.add_argument('--cost_limit', default=20., type=float)
    parser.add_argument('--use_softplus', default=True, type=boolean)
    parser.add_argument('--use_mlp_multiplier', default=True, type=boolean) # cbf need value below zero



    return parser

if __name__ == '__main__':
    args = get_parser().parse_args() 
    run(vars(args))
