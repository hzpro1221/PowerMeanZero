from ding.policy import create_policy
from easydict import EasyDict
from ding.config import compile_config
from zoo.board_games.alphabeta_pruning_bot import AlphaBetaPruningBot
from zoo.board_games.tictactoe.envs.tictactoe_env import TicTacToeEnv
import time
import torch 
import numpy as np
import copy

def config_policy_env():
    # ==============================================================
    # begin of the most frequently changed config specified by the user
    # ==============================================================
    collector_env_num = 8
    n_episode = 8
    evaluator_env_num = 5
    num_simulations = 25
    update_per_collect = 50
    batch_size = 256
    max_env_step = 20000
    mcts_ctree = True
    # ==============================================================
    # end of the most frequently changed config specified by the user
    # ==============================================================
    tictactoe_alphazero_config = dict(
        exp_name='data_az_ctree/tictactoe_sp-mode_alphazero_seed0',
        env=dict(
            board_size=3,
            battle_mode='self_play_mode',
            bot_action_type='alpha_beta_pruning',  # {'v0', 'alpha_beta_pruning'}
            channel_last=False,
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
            n_evaluator_episode=evaluator_env_num,
            manager=dict(shared_memory=False, ),
            # ==============================================================
            # for the creation of simulation env
            agent_vs_human=False,
            prob_random_agent=0,
            prob_expert_agent=0,
            scale=True,
            alphazero_mcts_ctree=mcts_ctree,
            save_replay_gif=False,
            replay_path_gif='./replay_gif',
            # ==============================================================
        ),
        policy=dict(
            mcts_ctree=mcts_ctree,
            # ==============================================================
            # for the creation of simulation env
            simulation_env_id='tictactoe',
            simulation_env_config_type='self_play',
            # ==============================================================
            model=dict(
                observation_shape=(3, 3, 3),
                action_space_size=int(1 * 3 * 3),
                # We use the small size model for tictactoe.
                num_res_blocks=1,
                num_channels=16,
                value_head_hidden_channels=[8],
                policy_head_hidden_channels=[8],
            ),
            cuda=True,
            board_size=3,
            update_per_collect=update_per_collect,
            batch_size=batch_size,
            optim_type='Adam',
            piecewise_decay_lr_scheduler=False,
            learning_rate=0.003,
            grad_clip_value=0.5,
            value_weight=1.0,
            entropy_weight=0.0,
            n_episode=n_episode,
            eval_freq=int(2e3),
            mcts=dict(num_simulations=num_simulations),
            collector_env_num=collector_env_num,
            evaluator_env_num=evaluator_env_num,
        ),
    )

    tictactoe_alphazero_config = EasyDict(tictactoe_alphazero_config)
    main_config = tictactoe_alphazero_config

    tictactoe_alphazero_create_config = dict(
        env=dict(
            type='tictactoe',
            import_names=['zoo.board_games.tictactoe.envs.tictactoe_env'],
        ),
        env_manager=dict(type='subprocess'),
        policy=dict(
            type='alphazero',
            import_names=['lzero.policy.alphazero'],
        ),
        collector=dict(
            type='episode_alphazero',
            import_names=['lzero.worker.alphazero_collector'],
        ),
        evaluator=dict(
            type='alphazero',
            import_names=['lzero.worker.alphazero_evaluator'],
        )
    )
    tictactoe_alphazero_create_config = EasyDict(tictactoe_alphazero_create_config)
    create_config = tictactoe_alphazero_create_config
    return main_config, create_config

def config_alpha_beta_prunning():
    cfg = dict(
        prob_random_agent=0,
        prob_expert_agent=0,
        battle_mode='self_play_mode',
        agent_vs_human=False,
        bot_action_type='alpha_beta_pruning',  # {'v0', 'alpha_beta_pruning'}
        channel_last=False,
        scale=True,
    )
    return cfg

def config_tree():
    tree_config=dict(
        # (int) The number of simulations to perform at each move.
        num_simulations=100,
        # (int) The maximum number of moves to make in a game.
        max_moves=512,  # for chess and shogi, 722 for Go.
        # (float) The alpha value used in the Dirichlet distribution for exploration at the root node of the search tree.
        root_dirichlet_alpha=0.3,
        # (float) The noise weight at the root node of the search tree.
        root_noise_weight=0.25,
        # (int) The base constant used in the PUCT formula for balancing exploration and exploitation during tree search.
        pb_c_base=19652,
        # (float) The initialization constant used in the PUCT formula for balancing exploration and exploitation during tree search.
        pb_c_init=1.25,
    )
    tree_config = EasyDict(tree_config)
    return tree_config

def mock_policy_value_func(env):
    # For mocking purposes, we return a distribution of 1 over legal actions
    action_probs_dict = {}
    legal_actions = env.legal_actions
    for action in legal_actions:
        action_probs_dict[action] = 1.0

    # Get roll-out value
    num_rollout = 100

    # Get environment observation
    action_mask = np.zeros(env.total_num_actions, 'int8')
    action_mask[env.legal_actions] = 1
    state_config_for_env_reset = {
        'init_state': copy.deepcopy(env.board),
        'start_player_index': env.players.index(env.current_player),
        'katago_policy_init': False,
        'katago_game_state': None
    }
    total_reward = 0.0    
    
    # Perform rollouts
    for _ in range(num_rollout):
        env.reset(
            start_player_index=state_config_for_env_reset.get('start_player_index'),
            init_state=state_config_for_env_reset.get('init_state')
        )
        env.battle_mode = env.battle_mode_in_simulation_env
        while not env.get_done_winner()[0]:
            # Randomly pick action
            action = np.random.choice(env.legal_actions)
            env.step(action)
        
        # Get the leaf value            
        done, winner = env.get_done_winner()
        if (winner == -1):
            leaf_value = 0
        else:
            leaf_value = -1 if state_config_for_env_reset.get('start_player_index') == winner else 1
        total_reward += leaf_value
    # Average the total reward over the number of rollouts
    avr_reward = total_reward / num_rollout 
    
    # Reset the evironment
    env.reset(
        start_player_index=state_config_for_env_reset.get('start_player_index'),
        init_state=state_config_for_env_reset.get('init_state')
    )
    return action_probs_dict, avr_reward

if __name__ == "__main__":
    # Path to AlphaZero with UCT agorithm
    alpha_zero_uct_path = []

    # Path to AlphaZero with Stochastic PowerMean UCT algorithm
    alpha_zero_powermean_uct_path = [] 

    # Prepare config for policy and environment
    cfg, create_cfg = config_policy_env()
    cfg = compile_config(cfg, seed=0, env=None, auto=True, create_cfg=create_cfg, save_cfg=True)

    # Loading policy and checkpoint
    policy = create_policy(cfg.policy, model=None, enable_field=['learn', 'collect', 'eval'])
    # policy.learn_mode.load_state_dict(torch.load(alpha_zero_uct_path[0], map_location=cfg.policy.device))

    # Alpha-Beta prunning algorithm
    cfg_alpha_beta = config_alpha_beta_prunning()
    alpha_beta_prunning_agorithm = AlphaBetaPruningBot(TicTacToeEnv, cfg_alpha_beta, 'alpha_beta_pruning_player')

    # Preparing environment
    env = TicTacToeEnv(EasyDict(cfg_alpha_beta))
    env_for_UCT = TicTacToeEnv(EasyDict(cfg_alpha_beta))
    env_for_stochastc_powermean_UCT = TicTacToeEnv(EasyDict(cfg_alpha_beta))

    tree_config= config_tree()
    # UCT agorithm
    from lzero.mcts.ptree.ptree_az import MCTS as mcts_uct 
    UCT = mcts_uct(tree_config, env_for_UCT)

    # Stochastic PowerMean UCT algorithm
    from lzero.mcts.ctree.ctree_alphazero.test.eval_alphazero_ctree import find_and_add_to_sys_path
    # Use the function to add the desired path to sys.path
    find_and_add_to_sys_path("lzero/mcts/ctree/ctree_alphazero/build")
    import mcts_alphazero
    stochastc_powermean_UCT = mcts_alphazero.MCTS(tree_config.max_moves, tree_config.num_simulations,
                                                tree_config.pb_c_base,
                                                tree_config.pb_c_init, tree_config.root_dirichlet_alpha,
                                                tree_config.root_noise_weight, env_for_stochastc_powermean_UCT)            

    # Test code for the tournament between UCT and stochastc_powermean_UCT
    player_index = 0
    env.reset()
    while not env.get_done_reward()[0]:
        state_config_for_env_reset = {
            'init_state': copy.deepcopy(env.board),
            'start_player_index': env.players.index(env.current_player),
            'katago_policy_init': False,
            'katago_game_state': None
        }
        print("start_player_index:", state_config_for_env_reset.get('start_player_index'))

        if (player_index == 0):
            start = time.time()
            action = UCT.get_next_action(state_config_for_env_reset, mock_policy_value_func, 1.0, False)
            print('UCT action time: ', time.time() - start)
            player_index = 1
        else:
            start = time.time()
            action = stochastc_powermean_UCT.get_next_action(state_config_for_env_reset, mock_policy_value_func, 1.0, False)
            print('Stochastic PowerMean UCT action time: ', time.time() - start)
            player_index = 0
        env.step(action[0])
        print("state:", env.board)
        print('-' * 15)
    print('reward: ', env.get_done_reward()[0])

    # # Hyperparameter for the tournament
    # number_of_game = 100
    
    # for _ in range(number_of_game):
    #     # Reset environment and player index
    #     env.reset()
    #     player_index = 0  # player 1 fist

    #     while not env.get_done_reward()[0]:
    #             if player_index == 0:
    #                 start = time.time()
    #                 action = player_0.get_best_action(state, player_index=player_index)
    #                 print('player 1 action time: ', time.time() - start)
    #                 player_index = 1
    #             else:
    #                 start = time.time()
    #                 action = player_1.get_best_action(state, player_index=player_index)
    #                 print('player 2 action time: ', time.time() - start)
    #                 player_index = 0
    #             env.step(action)
    #             state = env.board
    #             print('-' * 15)
    #             print(state)
    #             row, col = env.action_to_coord(action)        