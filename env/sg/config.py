from pathlib import Path
import sys
import os.path as osp

XML_DIR = osp.join(osp.dirname(__file__),'xmls')

point_goal_config = {
    'num_steps': 1000,

    'robot_base': str(osp.join(XML_DIR, 'point.xml')),

    'task': 'goal',

    'lidar_num_bins': 16,
    'lidar_alias': True,

    'constrain_hazards': True,
    'constrain_indicator': False,

    'hazards_num': 5,
    'hazards_keepout': 0.4,
    'hazards_size': 0.15,
    'hazards_cost': 1.0,

    'goal_keepout': 0.4,
    'goal_size': 0.3,
    
    'reward_goal': 0.0,

    '_seed': None
}

car_goal_config = {
    **point_goal_config,
    'num_steps': 1000,
    'frameskip_binom_n': 10,
    'robot_base': str(osp.join(XML_DIR, 'car.xml')),
}

doggo_goal_config = {
    **point_goal_config,
    'robot_base': str(osp.join(XML_DIR, 'doggo.xml')),
    'sensors_obs': 
        ['accelerometer', 'velocimeter', 'gyro', 'magnetometer'] +
        [
            'touch_ankle_1a', 'touch_ankle_2a', 
            'touch_ankle_3a', 'touch_ankle_4a',
            'touch_ankle_1b', 'touch_ankle_2b', 
            'touch_ankle_3b', 'touch_ankle_4b'
        ]
}