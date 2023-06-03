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
from run_sacl import SACL

import wandb
from tqdm import tqdm
from network import Critic, V_critic

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, device):
        self.max_size = int(1e6)
        self.device = device
        self.count = 0
        self.size = 0
        self.s = np.zeros((self.max_size, state_dim))
        self.a = np.zeros((self.max_size, action_dim))
        self.r = np.zeros((self.max_size, 1))
        self.s_ = np.zeros((self.max_size, state_dim))
        self.dw = np.zeros((self.max_size, 1))
        self.c = np.zeros((self.max_size, 1)) # cost
        self.cv = np.zeros((self.max_size, 1)) # constraint violation

        self.ss = np.zeros((self.max_size, state_dim)) # safe state
        self.ss_count = 0
        self.ss_size = 0
        self.us = np.zeros((self.max_size, state_dim)) # unsafe state
        self.us_count = 0
        self.us_size = 0


    def store(self, s, a, r, s_, dw, c, cv, ss=None, us=None):
        self.s[self.count] = s
        self.a[self.count] = a
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.c[self.count] = c
        self.cv[self.count] = cv      
        self.count = (self.count + 1) % self.max_size  # When the 'count' reaches max_size, it will be reset to 0.
        self.size = min(self.size + 1, self.max_size)  # Record the number of  transitions
    
    def store_safe_init(self, ss):
        self.ss[self.ss_count] = ss
        self.ss_count = (self.ss_count + 1) % self.max_size
        self.ss_size = min(self.ss_size + 1, self.max_size)

    def store_unsafe_state(self, us):
        self.us[self.us_count] = us
        self.us_count = (self.us_count + 1) % self.max_size
        self.us_size = min(self.us_size + 1, self.max_size)

    def sample(self, batch_size):
        index = np.random.choice(self.size, size=batch_size)  # Randomly sampling
        batch_s = torch.tensor(self.s[index], dtype=torch.float).to(self.device)
        batch_a = torch.tensor(self.a[index], dtype=torch.float).to(self.device)
        batch_r = torch.tensor(self.r[index], dtype=torch.float).to(self.device)
        batch_s_ = torch.tensor(self.s_[index], dtype=torch.float).to(self.device)
        batch_dw = torch.tensor(self.dw[index], dtype=torch.float).to(self.device)
        batch_c = torch.tensor(self.c[index], dtype=torch.float).to(self.device)
        batch_cv = torch.tensor(self.cv[index], dtype=torch.float).to(self.device)

        ss_index = np.random.choice(self.ss_size, size=batch_size)
        batch_ss = torch.tensor(self.ss[ss_index], dtype=torch.float).to(self.device)

        us_index = np.random.choice(self.us_size, size=batch_size)
        batch_us = torch.tensor(self.us[us_index], dtype=torch.float).to(self.device)

        return batch_s, batch_a, batch_r, batch_s_, batch_dw, batch_c, batch_cv, batch_ss, batch_us

    
class SAC_CBF(SACL):
    def __init__(self, state_dim, action_dim, max_action, config):
        super().__init__(state_dim, action_dim, max_action, config)

        self.cbf_lambda = config['cbf_lambda']

        self.certificate = V_critic(state_dim, self.hidden_width).to(self.device)
        self.certificate_optimizer = torch.optim.Adam(self.certificate.parameters(), lr=self.lr)
        self.certificate_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.certificate_optimizer,
            T_max=self.updates_per_training,
            eta_min=self.lr_end
        )

        # single q net
        self.cost_critic = Critic(state_dim, action_dim, self.hidden_width, use_softplus=config['use_softplus']).to(self.device)
        self.cost_critic_optimizer = torch.optim.Adam(self.cost_critic.parameters(), lr=self.lr)
        self.cost_critic_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.cost_critic_optimizer,
            T_max=self.updates_per_training,
            eta_min=self.lr_end
        )

    def update_cost_certificate(self, batch_s, batch_s_, batch_ss, batch_us, result):
        feasible_loss = torch.mean(torch.relu(self.certificate(batch_ss)))
        infeasible_loss = torch.mean(torch.relu(-self.certificate(batch_us)))
        invariance = self.certificate(batch_s_) - (1 - self.cbf_lambda) *self.certificate(batch_s)
        invariant_loss = torch.mean(torch.relu(invariance))
        cbf_loss = feasible_loss + infeasible_loss + invariant_loss
        result.update({
            'feasible_loss': feasible_loss,
            'infeasible_loss': infeasible_loss,
            'invariant_loss': invariant_loss,
            'cbf_loss': cbf_loss,
            'certificate': self.certificate(batch_s).detach().mean(),
        })
        return result
        
    def update_cost_critic(self, batch_s, batch_a, batch_s_, result): # not need batch_r！

        with torch.no_grad():
            target_Qc = self.certificate(batch_s_) - (1 - self.cbf_lambda) *self.certificate(batch_s)

        # Compute current Qc
        current_Qc = self.cost_critic(batch_s, batch_a)
        # Compute cost_critic loss
        cost_critic_loss = F.mse_loss(current_Qc, target_Qc)

        result.update({
            'cost_critic_loss': cost_critic_loss,
            'current_qc': current_Qc.detach().mean(),
            'target_qc': target_Qc.detach().mean(),
        })
        return result
    
    def learn(self, batch_s, batch_a, batch_r, batch_s_, batch_dw, batch_c, batch_cv, batch_ss, batch_us, result={}):
        # Compute critic loss
        result = self.update_critic(batch_s, batch_a, batch_r, batch_s_, batch_dw, result)
        result = self.update_cost_certificate(batch_s, batch_s_, batch_ss, batch_us, result)
        result = self.update_cost_critic(batch_s, batch_a, batch_s_, result)
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        result['critic_loss'].backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm)
        self.critic_optimizer.step()
        self.critic_lr_scheduler.step()

        self.certificate_optimizer.zero_grad()
        result['cbf_loss'].backward()
        torch.nn.utils.clip_grad_norm_(self.certificate.parameters(), max_norm=self.grad_norm)
        self.certificate_optimizer.step()
        self.certificate_lr_scheduler.step()

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
        # self.soft_update(self.cost_critic, self.cost_critic_target)

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

    agent = SAC_CBF(state_dim, action_dim, max_action, config)
    replay_buffer = ReplayBuffer(state_dim, action_dim, config['device'])

    evaluate_num = 0  # Record the number of evaluations
    total_steps = 0  # Record the total steps during the training
    red_list = ['ep_r', 'ep_c']
    result = {}

    while total_steps < config['max_train_steps']:
        s = env.reset()
        replay_buffer.store_safe_init(s) # assume initial state is safe
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
            
            if c == 1.:
                replay_buffer.store_unsafe_state(s_)

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
    parser.add_argument('--algo', default='SAC_CBF', type=str)
    parser.add_argument('--environment', default='safety_gym', type=str) # simple_sg or safety_gym
    # parser.add_argument('--environment', default='simple_sg', type=str)
    parser.add_argument('--robot', default='Point', type=str)
    parser.add_argument('--task', default='Goal1', type=str)
    parser.add_argument('--env_name', default='point', type=str)
    parser.add_argument('--device', default='cuda:3', type=str)
    parser.add_argument('--wandb', default=False, type=boolean)
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--max_train_steps', default=int(2e6), type=int)
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
    parser.add_argument('--actor_update_interval', default=int(2), type=int)
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
    parser.add_argument('--multiplier_update_interval', default=int(5), type=int)
    parser.add_argument('--multiplier_init', default=0.5, type=float)
    parser.add_argument('--multiplier_ub', default=25., type=float)
    parser.add_argument('--penalty_ub', default=100., type=float)
    parser.add_argument('--penalty_lb', default=-1., type=float)
    parser.add_argument('--cost_limit', default=0., type=float) # cbf use Q(s,a)<=0

    parser.add_argument('--use_softplus', default=False, type=boolean) # cbf need value below zero
    parser.add_argument('--cbf_lambda', default=0.1, type=float) 



    return parser

if __name__ == '__main__':
    args = get_parser().parse_args() 
    run(vars(args))
