from easydict import EasyDict

dmc_state_env_action_space_map = EasyDict({
    'acrobot-swingup':1,
    'cartpole-balance': 1,
    'cartpole-balance_sparse': 1,
    'cartpole-swingup': 1,
    'cartpole-swingup_sparse': 1,
    'cheetah-run': 6,
    "ball_in_cup-catch":2,
    "finger-spin":2,
    "finger-turn_easy":2,
    "finger-turn_hard":2,
    'hopper-hop': 4,
    'hopper-stand': 4,
    'pendulum-swingup': 1,
    'quadruped-run': 12,
    'quadruped-walk': 12,
    'reacher-easy': 2,
    'reacher-hard':2,
    'walker-run': 6,
    'walker-stand': 6,
    'walker-walk': 6,
    'humanoid-run': 21,

})

dmc_state_env_obs_space_map = EasyDict({
    'acrobot-swingup':6,
    'cartpole-balance': 5,
    'cartpole-balance_sparse': 5,
    'cartpole-swingup': 5,
    'cartpole-swingup_sparse': 5,
    'cheetah-run': 17,
    "ball_in_cup-catch":8,
    "finger-spin":9,
    "finger-turn_easy":12,
    "finger-turn_hard":12,
    'hopper-hop': 15,
    'hopper-stand': 15,
    'pendulum-swingup': 3,
    'quadruped-run': 78,
    'quadruped-walk': 78,
    'reacher-easy': 6,
    'reacher-hard':6,
    'walker-run': 24,
    'walker-stand': 24,
    'walker-walk': 24,
    'humanoid-run': 67,
})