from easydict import EasyDict

# ==============================================================
# begin of the most frequently changed config specified by the user
# ==============================================================
collector_env_num = 8
n_episode = 8
evaluator_env_num = 3
num_simulations = 50
update_per_collect = 200
batch_size = 256
max_env_step = int(1e6)
reanalyze_ratio = 0
# ==============================================================
# end of the most frequently changed config specified by the user
# ==============================================================

cfg = dict(
    main_config=dict(
        exp_name='CartPole-v0-MuZero',
        seed=0,
        env=dict(
            env_id='Pendulum-v1',
            continuous=False,
            manually_discretization=True,
            each_dim_disc_size=11,
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            n_evaluator_episode=evaluator_env_num,
            manager=dict(shared_memory=False, ),
        ),
        policy=dict(
            model=dict(
                observation_shape=3,
                action_space_size=11,
                model_type='mlp', 
                lstm_hidden_size=128,
                latent_state_dim=128,
                self_supervised_learning_loss=True,  # NOTE: default is False.
            ),
            cuda=True,
            env_type='not_board_games',
            game_segment_length=50,
            update_per_collect=update_per_collect,
            batch_size=batch_size,
            optim_type='Adam',
            piecewise_decay_lr_scheduler=False,
            learning_rate=0.003,
            ssl_loss_weight=2,  # NOTE: default is 0.
            num_simulations=num_simulations,
            reanalyze_ratio=reanalyze_ratio,
            n_episode=n_episode,
            eval_freq=int(2e3),
            replay_buffer_size=int(1e6),  # the size/capacity of replay_buffer, in the terms of transitions.
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
        ),
        wandb_logger=dict(
            gradient_logger=False, video_logger=False, plot_logger=False, action_logger=False, return_logger=False
        ),
    ),
    create_config=dict(
        env=dict(
            type='pendulum_lightzero',
            import_names=['zoo.classic_control.pendulum.envs.pendulum_lightzero_env'],
        ),
        env_manager=dict(type='subprocess'),
        policy=dict(
            type='muzero',
            import_names=['lzero.policy.muzero'],
        ),
    ),
)

cfg = EasyDict(cfg)
