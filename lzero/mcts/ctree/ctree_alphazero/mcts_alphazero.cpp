#include "node_alphazero.h"
#include <cmath>
#include <map>
#include <random>
#include <list>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <opencv2/opencv.hpp>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <iomanip>

namespace py = pybind11;

// The MCTS class implements Monte Carlo Tree Search (MCTS) for AlphaZero-like algorithms.
class MCTS {

    private:
    int max_moves;                  // Maximum allowed moves in a game
    int num_simulations;            // Number of MCTS simulations
    double pb_c_base;               // Coefficient for UCB exploration term (base)
    double pb_c_init;               // Coefficient for UCB exploration term (initial value)
    double root_dirichlet_alpha;    // Alpha parameter for Dirichlet noise
    double root_noise_weight;       // Weight for exploration noise added to root node
    py::object simulate_env;        // Python object representing the simulation environment

    double c;                       // Explore constant
    double p;                       // Power value
    double gamma;                   // Gamma constant

    public:
    // Constructor to initialize MCTS with optional parameters
    MCTS(int max_moves=512, int num_simulations=800,
         double pb_c_base=19652, double pb_c_init=1.25,
         double root_dirichlet_alpha=0.3, 
         double root_noise_weight=0.25, 
         py::object simulate_env=py::none(),
         double c=1.14,
         double p=2,
         double gamma=0.99)
        : max_moves(max_moves), num_simulations(num_simulations),
          pb_c_base(pb_c_base), pb_c_init(pb_c_init),
          root_dirichlet_alpha(root_dirichlet_alpha),
          root_noise_weight(root_noise_weight),
          simulate_env(simulate_env),
          c(c),
          p(p),
          gamma(gamma) {}

    // Getter for the simulation environment (Python object)
    py::object get_simulate_env() const {
        return simulate_env;
    }

    // Setter for the simulation environment
    void set_simulate_env(py::object env) {
        simulate_env = env;
    }

    // Getter for pb_c_base
    double get_pb_c_base() const { return pb_c_base; }

    // Getter for pb_c_init
    double get_pb_c_init() const { return pb_c_init; }

    // Calculate the Upper Confidence Bound score of Stochastic-Power-UCT for child nodes
    double _ucb_score(int opp_mov, std::shared_ptr<Node> V_sh, std::shared_ptr<Node> Q_sh_a) {
        // Calculate PB-C component of UCB
        double pb_c = c * std::sqrt(std::sqrt(V_sh->visit_count)) / std::sqrt(Q_sh_a->visit_count);

        // Combine prior probability and value score
        double prior_score = pb_c * Q_sh_a->prior_p[opp_mov];
        double value_score = Q_sh_a->value;
        return prior_score + value_score;
    }

    // Add Dirichlet noise to the root node for exploration
    void _add_exploration_noise(int opp_mov, std::shared_ptr<Node> node) {
        std::vector<int> actions;
        // Collect all child actions of the root node
        for (const auto& kv : node->children) {
            actions.push_back(kv.first);
        }

        // Generate Dirichlet noise
        std::default_random_engine generator;
        std::gamma_distribution<double> distribution(root_dirichlet_alpha, 1.0);

        std::vector<double> noise;
        double sum = 0;
        for (size_t i = 0; i < actions.size(); ++i) {
            double sample = distribution(generator);
            noise.push_back(sample);
            sum += sample;
        }

        // Normalize the noise
        for (size_t i = 0; i < noise.size(); ++i) {
            noise[i] /= sum;
        }

        // Mix noise with prior probabilities
        double frac = root_noise_weight;
        for (size_t i = 0; i < actions.size(); ++i) {
            node->children[actions[i]]->prior_p[-1] = node->children[actions[i]]->prior_p[opp_mov] * (1 - frac) + noise[i] * frac;
        }
    }

    // Select the best child node based on UCB score
    std::tuple<int, std::shared_ptr<Node>, std::shared_ptr<Node>> _select_child(int opp_mov, std::shared_ptr<Node> V_sh, std::shared_ptr<Node> Q_sh, py::object simulate_env, py::object policy_value_func) {
        py::print("\nSelect child with opp_mov: ", opp_mov);
        py::print("Before selecting child");
        for (const auto& kv : Q_sh->children) {
            int action_tmp = kv.first;
            std::shared_ptr<Node> Q_sh_a_tmp = kv.second;
            py::print("Action: ", action_tmp, " Prior P: ", Q_sh_a_tmp->prior_p[opp_mov], " Init Reward: ", Q_sh_a_tmp->init_reward);
        }
        
        int action = -1;
        std::shared_ptr<Node> V_sh_plus_1 = nullptr;
        std::shared_ptr<Node> Q_sh_a = nullptr;
        double best_score = -9999999;
        int policy_func_called = 0;

        py::tuple result;
        std::map<int, double> action_probs_dict;
        double leaf_value;
        py::list legal_actions_list;
        std::vector<int> legal_actions;

        // If opp_mov has not been explored yet
        if (std::find(Q_sh->opp_mov.begin(), Q_sh->opp_mov.end(), opp_mov) == Q_sh->opp_mov.end()) {
            // Get action probabilities and leaf value from the policy-value function
            result = policy_value_func(simulate_env);
            action_probs_dict = result[0].cast<std::map<int, double>>();
            leaf_value = result[1].cast<double>();                

            // Extract legal actions from the simulation environment
            legal_actions_list = simulate_env.attr("legal_actions").cast<py::list>();
            legal_actions = legal_actions_list.cast<std::vector<int>>();
            
            // Mark as policy function called 
            policy_func_called = 1;
            
            for (const auto& kv : action_probs_dict) {
                int action = kv.first;
                double prior_p = kv.second;
                
                // Check if the action is legal
                if (std::find(legal_actions.begin(), legal_actions.end(), action) != legal_actions.end()) {
                    // If the child node for the action already exists, update its prior
                    if (Q_sh->children.find(action) != Q_sh->children.end()) {
                        Q_sh->children[action]->prior_p[opp_mov] = prior_p; 
                    } else {
                        // Create new child nodes for both V and Q trees
                        V_sh->children[action] = std::make_shared<Node>(V_sh);
                        Q_sh->children[action] = std::make_shared<Node>(Q_sh);
                                                    
                        Q_sh->children[action]->prior_p[opp_mov] = prior_p;
                    }
                }
            }

            // Mark opp_mov as explored
            Q_sh->opp_mov.push_back(opp_mov);            
        }        

        legal_actions_list = simulate_env.attr("legal_actions").cast<py::list>();
        legal_actions = legal_actions_list.cast<std::vector<int>>();

        // For each legal action, check if it has not been expanded yet
        for (size_t i = 0; i < legal_actions.size(); ++i) {
            int action = legal_actions[i];

            // If the action has not been expanded, initialize its child nodes
            if (Q_sh->children.find(action) == Q_sh->children.end()) {
                if (policy_func_called == 0) {
                    // Get action probabilities and leaf value from the policy-value function
                    result = policy_value_func(simulate_env);
                    action_probs_dict = result[0].cast<std::map<int, double>>();
                    leaf_value = result[1].cast<double>();

                    // Mark as policy function called 
                    policy_func_called = 1;                            
                }

                // Create new child nodes for both V and Q trees
                V_sh->children[action] = std::make_shared<Node>(V_sh);
                Q_sh->children[action] = std::make_shared<Node>(Q_sh);

                // Set prior probability for the Q node (based on the opponent move)
                Q_sh->children[action]->prior_p[opp_mov] = action_probs_dict[action];
            }
        }

        // Iterate over all child nodes of Q_sh
        for (const auto& kv : Q_sh->children) {
            int action_tmp = kv.first;
            std::shared_ptr<Node> Q_sh_a_tmp = kv.second;

            // Check if the current action is legal before proceeding
            if (std::find(legal_actions.begin(), legal_actions.end(), action_tmp) != legal_actions.end()) {
                // If prior probability for the current opponent move is uninitialized, initialize it
                if (Q_sh_a_tmp->prior_p[opp_mov] == 0) {
                    if (policy_func_called == 0) {
                        // Get action probabilities and leaf value from the policy-value function
                        result = policy_value_func(simulate_env);
                        action_probs_dict = result[0].cast<std::map<int, double>>();
                        leaf_value = result[1].cast<double>();
    
                        // Mark as policy function called 
                        policy_func_called = 1;                            
                    }

                    // Assign prior probability if available
                    if (action_probs_dict.find(action_tmp) != action_probs_dict.end()) {
                        Q_sh_a_tmp->prior_p[opp_mov] = action_probs_dict[action_tmp];
                    } else {
                        Q_sh_a_tmp->prior_p[opp_mov] = 0.0;  // Fallback to 0 if action not found
                    }
                }

                // Compute UCB score for the current action
                double score = _ucb_score(opp_mov, V_sh, Q_sh_a_tmp);

                // Update best action if this one has the highest score so far
                if (score > best_score) {
                    best_score = score;
                    action = action_tmp;
                    V_sh_plus_1 = V_sh->children[action_tmp];
                    Q_sh_a = Q_sh->children[action_tmp];
                }
            }
        }
        
        // If no valid child node was selected, fallback to the current node
        if ((V_sh_plus_1 == nullptr) && (Q_sh_a == nullptr)) {
            V_sh_plus_1 = V_sh;
            Q_sh_a = Q_sh;
        }

        py::print("action_probs_dict: ", action_probs_dict);
        py::print("After selecting child");
        for (const auto& kv : Q_sh->children) {
            int action_tmp = kv.first;
            std::shared_ptr<Node> Q_sh_a_tmp = kv.second;
            py::print("Action: ", action_tmp, " Prior P: ", Q_sh_a_tmp->prior_p[opp_mov], " Init Reward: ", Q_sh_a_tmp->init_reward);
        }

        return std::make_tuple(action, V_sh_plus_1, Q_sh_a);
    }

    // Expand a leaf node by adding its legal child actions based on policy network output
    double _expand_leaf_node(int opp_mov, std::shared_ptr<Node> V_sh, std::shared_ptr<Node> Q_sh, py::object simulate_env, py::object policy_value_func) {
        std::map<int, double> action_probs_dict;
        double leaf_value;
        py::print("\nExpanding leaf node with opp_mov: ", opp_mov);
        py::print("action_probs_dict: ", action_probs_dict);
        // Query the policy-value function to obtain action probabilities and a leaf value estimate
        py::tuple result = policy_value_func(simulate_env);
        action_probs_dict = result[0].cast<std::map<int, double>>();
        leaf_value = result[1].cast<double>();

        // Retrieve legal actions from the simulation environment
        py::list legal_actions_list = simulate_env.attr("legal_actions").cast<py::list>();
        std::vector<int> legal_actions = legal_actions_list.cast<std::vector<int>>();

        // Create child nodes for each legal action and assign prior probabilities
        for (const auto& kv : action_probs_dict) {
            int action = kv.first;
            double prior_p = kv.second;
            if (std::find(legal_actions.begin(), legal_actions.end(), action) != legal_actions.end()) {
                // Create new child nodes for both V and Q trees
                V_sh->children[action] = std::make_shared<Node>(V_sh);
                Q_sh->children[action] = std::make_shared<Node>(Q_sh);

                // Set prior probability for the Q node (based on the opponent move)
                Q_sh->children[action]->prior_p[opp_mov] = action_probs_dict[action];
            }
        }

        // Record the opponent move and initialize the reward estimate for the current node
        Q_sh->opp_mov.push_back(opp_mov);
        Q_sh->init_reward = leaf_value;
        
        py::print("After expanding leaf node");
        for (const auto& kv : Q_sh->children) {
            int action_tmp = kv.first;
            std::shared_ptr<Node> Q_sh_a_tmp = kv.second;
            py::print("Action: ", action_tmp, " Prior P: ", Q_sh_a_tmp->prior_p[opp_mov], " Init Reward: ", Q_sh_a_tmp->init_reward);
        }
        return leaf_value;
    }

    // Perform MCTS to select the next action
    std::tuple<int, std::vector<double>, std::shared_ptr<Node>> get_next_action(py::object state_config_for_env_reset, py::object policy_value_func, double temperature, bool sample) {
        std::shared_ptr<Node> V_root = std::make_shared<Node>();
        std::shared_ptr<Node> Q_root = std::make_shared<Node>();
        std::shared_ptr<Node> V_root_2 = std::make_shared<Node>();
        std::shared_ptr<Node> Q_root_2 = std::make_shared<Node>();                

        // Reset the environment to the initial state
        py::object init_state = state_config_for_env_reset["init_state"];
        if (!init_state.is_none()) {
            init_state = py::bytes(init_state.attr("tobytes")());
        }
        py::object katago_game_state = state_config_for_env_reset["katago_game_state"];
        if (!katago_game_state.is_none()) {
            katago_game_state = py::module::import("pickle").attr("dumps")(katago_game_state);
        }

        simulate_env.attr("reset")(
            state_config_for_env_reset["start_player_index"].cast<int>(),
            init_state,
            state_config_for_env_reset["katago_policy_init"].cast<bool>(),
            katago_game_state
        );

        // Expand the root node
        double leaf_value = _expand_leaf_node(-1, V_root, Q_root, simulate_env, policy_value_func);
        V_root->visit_count++;
        V_root->value = leaf_value;
        V_root->flag = -1;

        // Add Dirichlet noise to encourage exploration if sampling
        if (sample) {
            _add_exploration_noise(-1, V_root);
            _add_exploration_noise(-1, Q_root);
        }

        // Get legal actions
        // py::list legal_actions_py = simulate_env.attr("legal_actions").cast<py::list>();
        // std::vector<int> legal_actions;
        // for (py::handle h : legal_actions_py) {
        //     legal_actions.push_back(h.cast<int>());
        // }

        // Perform multiple MCTS simulations
        for (int n = 0; n < num_simulations; ++n) {
            simulate_env.attr("reset")(
                state_config_for_env_reset["start_player_index"].cast<int>(),
                init_state,
                state_config_for_env_reset["katago_policy_init"].cast<bool>(),
                katago_game_state
            );
            simulate_env.attr("battle_mode") = simulate_env.attr("battle_mode_in_simulation_env");

            _simulateV(-1, V_root, Q_root, V_root_2, Q_root_2, simulate_env, policy_value_func);

            // Export both trees to PNG via Graphviz
            exportToDot(-1, "/content/tree_player_1.dot", V_root, Q_root);
            std::string command = "dot -Tpng -Gdpi=300 /content/tree_player_1.dot -o /content/log_v2/tree_player_1_tree_legal_ac_len_" + std::to_string(legal_actions.size()) + "_num_sti_" + std::to_string(n) + ".png";
            std::system(command.c_str());        

            exportToDot(-1, "/content/tree_player_2.dot", V_root_2, Q_root_2);
            std::string command2 = "dot -Tpng -Gdpi=300 /content/tree_player_2.dot -o /content/log_v2/tree_player_2_tree_legal_ac_len_" + std::to_string(legal_actions.size()) + "_num_sti_" + std::to_string(n) + ".png";
            std::system(command2.c_str());                    
        }

        // Visualization for tic-tac-toe
        // drawTicTacToeBoard(legal_actions, Q_root, "/content/log_v2_2/tictactoe_board" + std::to_string(legal_actions.size()) + ".png");
        // drawTicTacToeBoardPriorP(legal_actions, -1, Q_root, "/content/log_v2_2/tictactoe_board_prior_p" + std::to_string(legal_actions.size()) + ".png");        

        // Count visits to each action
        std::vector<std::pair<int, int>> action_visits;
        for (int action = 0; action < simulate_env.attr("action_space").attr("n").cast<int>(); ++action) {
            int visits = Q_root->children.count(action) ? Q_root->children[action]->visit_count : 0;
            action_visits.emplace_back(action, visits);
        }

        std::vector<int> actions, visits;
        for (const auto& [a, v] : action_visits) {
            actions.push_back(a);
            visits.push_back(v);
        }

        std::vector<double> action_probs = visit_count_to_action_distribution(std::vector<double>(visits.begin(), visits.end()), temperature);

        int action_selected;
        if (sample) {
            action_selected = random_choice(actions, action_probs);
        }
        else {
            action_selected = actions[std::distance(action_probs.begin(), std::max_element(action_probs.begin(), action_probs.end()))];
        }

        return std::make_tuple(action_selected, action_probs, Q_root);
    }

    // Perform a simulation step starting from the given V-node
    // Selects a child node, recursively simulates Q-node dynamics,
    // then updates the V-node's visit count and value estimate accordingly.
    double _simulateV(int opp_mov, std::shared_ptr<Node> V_sh, std::shared_ptr<Node> Q_sh, 
                    std::shared_ptr<Node> V_sh_2, std::shared_ptr<Node> Q_sh_2, 
                    py::object simulate_env, py::object policy_value_func) {
        // Select the best child action and associated nodes based on the current state
        int action;
        std::shared_ptr<Node> V_sh_plus_1;
        std::shared_ptr<Node> Q_sh_a;
        std::tie(action, V_sh_plus_1, Q_sh_a) = _select_child(opp_mov, V_sh, Q_sh, simulate_env, policy_value_func);

        // Recursively simulate the Q-node after taking the selected action
        double leaf_value = _simulateQ(opp_mov, action, V_sh_plus_1, Q_sh_a, V_sh_2, Q_sh_2, simulate_env, policy_value_func);

        // Update visit count of the current V-node
        V_sh->visit_count++;

        // Compute the updated value of V-node using a p-norm aggregation over Q-node children values weighted by visit counts
        double tmp = 0.0;
        for (const auto& kv : Q_sh->children) {
            int action_tmp = kv.first;
            std::shared_ptr<Node> Q_sh_a_tmp = kv.second;
            tmp += (static_cast<double>(Q_sh_a_tmp->visit_count) / V_sh->visit_count) * pow(Q_sh_a_tmp->value, p);
        }
        V_sh->value = pow(tmp, 1/p);

        // Return the leaf value propagated from the simulation
        return leaf_value;
    }

    // Perform a Q-node simulation step by applying the selected action,
    // evaluating game termination or expanding the tree,
    // then updating node statistics accordingly.
    double _simulateQ(int opp_mov, int action, std::shared_ptr<Node> V_sh_plus_1, std::shared_ptr<Node> Q_sh_a, 
                    std::shared_ptr<Node> V_sh_2, std::shared_ptr<Node> Q_sh_2, 
                    py::object simulate_env, py::object policy_value_func) {
        
        // Apply the chosen action in the environment
        simulate_env.attr("step")(action);

        // Retrieve game termination status and winner
        py::tuple game_result = simulate_env.attr("get_done_winner")();
        bool done = game_result[0].cast<bool>();
        int winner = game_result[1].cast<int>();

        double leaf_value;

        if (!done) {
            if (V_sh_plus_1->is_leaf()) {
                // Expand leaf node and get its value estimate
                leaf_value = _expand_leaf_node(opp_mov, V_sh_plus_1, Q_sh_a, simulate_env, policy_value_func);

                // Rescale leaf_value to [0,1]
                leaf_value = std::max(0.0, (leaf_value + 1) / 2.0);

                // Update V_sh_plus_1 node statistics
                V_sh_plus_1->visit_count++;
                V_sh_plus_1->value = leaf_value;
                V_sh_plus_1->flag = 3;
            } else {
                // Continue simulating from the next V-node (opponent's turn)
                leaf_value = 1.0 - _simulateV(action, V_sh_2, Q_sh_2, V_sh_plus_1, Q_sh_a, simulate_env, policy_value_func);
                V_sh_plus_1->flag = -1;
            }
        } else {
            // Game has ended: assign terminal values
            if (winner == -1) {
                leaf_value = 0.0;
                V_sh_plus_1->flag = 0;
                V_sh_2->flag = 0;
            } else {
                // leaf_value = -1 if current player won, else 1
                leaf_value = (simulate_env.attr("current_player").cast<int>() == winner) ? -1.0 : 1.0;

                if (leaf_value == -1.0) {
                    V_sh_plus_1->flag = 1;
                    V_sh_2->flag = 2;
                } else {
                    V_sh_plus_1->flag = 2;
                    V_sh_2->flag = 1;
                }
            }

            // Rescale leaf_value to [0,1]
            leaf_value = std::max(0.0, (leaf_value + 1) / 2.0);
        }

        // Update Q-node value as a running average incorporating leaf value and discounted V-node value
        Q_sh_a->value = (Q_sh_a->value * Q_sh_a->visit_count + leaf_value + gamma * V_sh_plus_1->value) / (Q_sh_a->visit_count + 1);
        Q_sh_a->visit_count++;

        return leaf_value;
    }

    void writeNode(int action, FILE* file, std::shared_ptr<Node> V_root, std::shared_ptr<Node> Q_root, 
                std::unordered_map<std::shared_ptr<Node>, int>& nodeToId, int& nextId) {
        
        if (nodeToId.find(V_root) == nodeToId.end()) {
            nodeToId[V_root] = nextId++;
        }
        int nodeId = nodeToId[V_root];

        // Calculate a smoother fill color based on visit_count (for a more pleasant color scale)
        double max_visit = num_simulations;  // Set a reasonable max visit count to normalize the color scaling
        double norm_visit = std::min(1.0, (double)V_root->visit_count / max_visit);  // Normalize visit count

        // Create a gradient from light blue to dark blue based on visit_count
        int red = static_cast<int>(255 * (1 - norm_visit)); // Fade from blue to red as visit_count increases
        int green = static_cast<int>(255 * (1 - norm_visit)); // Keep green minimal for contrast
        int blue = static_cast<int>(255 * norm_visit); // Fade to blue as count increases

        char fillColorStr[16];
        // Convert to hexadecimal color code for fill color
        snprintf(fillColorStr, sizeof(fillColorStr), "#%02X%02X%02X", red, green, blue);
        
        // Write node information with fill color and smooth transitions
        double penwidth = 1.0 + (double)V_root->visit_count * 0.02; // Adjust thickness based on visit count
        if (V_root->flag == -1) {
            fprintf(file, "    %d [label=\"Ac: %d\nvisit:%.0f\", fontsize=8, width=0.6, height=0.6, fixedsize=true, "
                "penwidth=%.2f, color=\"black\", fillcolor=\"%s\", style=filled];\n",
                nodeId, action, (double)V_root->visit_count, penwidth, fillColorStr);
        } else if (V_root->flag == 0) {
            // For debugging: 0 = lose
            fprintf(file, "    %d [label=\"LOSE\\nAc: %d\\nvisit:%.0f\", fontsize=8, width=0.6, height=0.6, fixedsize=true, "
                "penwidth=%.2f, color=\"black\", fillcolor=\"#FFAAAA\", style=filled];\n",
                nodeId, action, (double)V_root->visit_count, penwidth);
    
        } else if (V_root->flag == 1) {
            // For debugging: 1 = draw
            fprintf(file, "    %d [label=\"DRAW\\nAc: %d\\nvisit:%.0f\", fontsize=8, width=0.6, height=0.6, fixedsize=true, "
                "penwidth=%.2f, color=\"black\", fillcolor=\"#AAAAFF\", style=filled];\n",
                nodeId, action, (double)V_root->visit_count, penwidth);
    
        } else if (V_root->flag == 2) {
            // For debugging: 2 = win
            fprintf(file, "    %d [label=\"WIN\\nAc: %d\\nvisit:%.0f\", fontsize=8, width=0.6, height=0.6, fixedsize=true, "
                "penwidth=%.2f, color=\"black\", fillcolor=\"#AAFFAA\", style=filled];\n",
                nodeId, action, (double)V_root->visit_count, penwidth);
    
        } else if (V_root->flag == 3) {
            // For debugging: 3 = extended
            fprintf(file, "    %d [label=\"EXT\\nAc: %d\\nvisit:%.0f\", fontsize=8, width=0.6, height=0.6, fixedsize=true, "
                "penwidth=%.2f, color=\"black\", fillcolor=\"#FFFFAA\", style=filled];\n",
                nodeId, action, (double)V_root->visit_count, penwidth);
        }

        for (const auto& kv : Q_root->children) {
            int action_tmp = kv.first;
            std::shared_ptr<Node> Q_sh_a_tmp = kv.second;

            if (V_root->children.find(action_tmp) == V_root->children.end()) {
                continue;
            }

            std::shared_ptr<Node> V_sh_plus_1 = V_root->children[action_tmp];

            if (nodeToId.find(V_sh_plus_1) == nodeToId.end()) {
                nodeToId[V_sh_plus_1] = nextId++;
            }

            // if (V_sh_plus_1->is_leaf() && (V_sh_plus_1->visit_count == 0)) {
            //     continue;
            // }

            int childId = nodeToId[V_sh_plus_1];

            double q_value = std::max(0.0, Q_sh_a_tmp->value);  // Clamp negative values to 0

            double penwidth = 2.0 + q_value;
            int red = std::min(255, static_cast<int>(q_value * 20));
            int green = 255 - red;
            int blue = 255 - red;

            char colorStr[16];
            snprintf(colorStr, sizeof(colorStr), "#%02X%02X%02X", red, green, blue);
            
            std::ostringstream prior_p_stream;
            for (const auto& [action, prob] : Q_sh_a_tmp->prior_p) {
                prior_p_stream << action << ": " << std::fixed << std::setprecision(2) << prob << "\\n";
            }
            std::string prior_p_str = prior_p_stream.str();
            
            // Write edge information
            fprintf(file, "    %d -> %d [label=\"Q_val: %.2f\nprior_p: %s\", penwidth=%.2f, color=\"%s\", style=bold];\n",
                nodeId, childId, Q_sh_a_tmp->value, prior_p_str.c_str(), penwidth, colorStr);

            // Recursively write child node
            writeNode(action_tmp, file, V_sh_plus_1, Q_sh_a_tmp, nodeToId, nextId);
        }
    }

    void exportToDot(int action, const std::string& filename, std::shared_ptr<Node> V_root, std::shared_ptr<Node> Q_root) {
        FILE* file = fopen(filename.c_str(), "w");
        if (!file) {
            std::cerr << "Failed to open file: " << filename << std::endl;
            return;
        }

        // Global graph settings for compact layout
        fprintf(file, "digraph MCTS {\n");
        fprintf(file, "    node [shape=circle, style=filled, fillcolor=lightgrey, fontsize=10, width=0.5, height=0.5, fixedsize=true];\n");
        fprintf(file, "    edge [fontsize=8];\n");

        std::unordered_map<std::shared_ptr<Node>, int> nodeToId;
        int nextId = 0;

        writeNode(action, file, V_root, Q_root, nodeToId, nextId);

        fprintf(file, "}\n");
        fclose(file);
    } 

    void drawTicTacToeBoard(
        std::vector<int> legal_actions,
        std::shared_ptr<Node> root,
        const std::string& filename
    ) {
        const int cellSize = 100;
        const int lineThickness = 4;
        const int imageSize = cellSize * 3;
    
        cv::Mat image(imageSize, imageSize, CV_8UC3, cv::Scalar(255, 255, 255)); // White background
    
        // Draw grid lines
        for (int i = 1; i < 3; ++i) {
            int pos = i * cellSize;
            cv::line(image, cv::Point(pos, 0), cv::Point(pos, imageSize), cv::Scalar(0, 0, 0), lineThickness);
            cv::line(image, cv::Point(0, pos), cv::Point(imageSize, pos), cv::Scalar(0, 0, 0), lineThickness);
        }
    
        // Draw symbols and confidence overlays
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                int symbol = row * 3 + col; // Correct symbol calculation
                int x = col * cellSize;
                int y = row * cellSize;
                cv::Point center(x + cellSize / 2, y + cellSize / 2);
    
                if (std::find(legal_actions.begin(), legal_actions.end(), symbol) != legal_actions.end()) {
                    // Get the reward from the root node
                    double reward = root->children[symbol]->init_reward;

                    // Normalize reward from [-1, 1] to [0, 1] for alpha
                    double confidence = (reward + 1.0) / 2.0;  // Map -1→0, 0→0.5, 1→1.0
    
                    // Draw translucent red overlay
                    cv::Mat roi = image(cv::Rect(x, y, cellSize, cellSize));
                    cv::Mat overlay(cellSize, cellSize, CV_8UC3, cv::Scalar(0, 0, 255)); // Red
                    cv::addWeighted(overlay, confidence, roi, 1.0 - confidence, 0, roi);
    
                    // Draw actual reward value as text (not normalized)
                    std::string text = cv::format("%.2f", reward);
                    cv::putText(image, text, cv::Point(x + 25, y + 60),
                                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
                } else {
                    cv::circle(image, center, 35, cv::Scalar(255, 0, 0), 4);
                }
            }
        }
    
        cv::imwrite(filename, image);
    }

    void drawTicTacToeBoardPriorP(
        std::vector<int> legal_actions,
        int opp_mov,
        std::shared_ptr<Node> root,
        const std::string& filename
    ) {
        const int cellSize = 100;
        const int lineThickness = 4;
        const int imageSize = cellSize * 3;
    
        cv::Mat image(imageSize, imageSize, CV_8UC3, cv::Scalar(255, 255, 255)); // White background
    
        // Draw grid lines
        for (int i = 1; i < 3; ++i) {
            int pos = i * cellSize;
            cv::line(image, cv::Point(pos, 0), cv::Point(pos, imageSize), cv::Scalar(0, 0, 0), lineThickness);
            cv::line(image, cv::Point(0, pos), cv::Point(imageSize, pos), cv::Scalar(0, 0, 0), lineThickness);
        }
    
        // Draw symbols and confidence overlays
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                int symbol = row * 3 + col; // Correct symbol calculation
                int x = col * cellSize;
                int y = row * cellSize;
                cv::Point center(x + cellSize / 2, y + cellSize / 2);
    
                if (std::find(legal_actions.begin(), legal_actions.end(), symbol) != legal_actions.end()) {
                    // Get the prior_p from the root node
                    double prior_p = root->children[symbol]->prior_p[opp_mov];

                    // Use prior_p directly as confidence (no normalization needed)
                    double confidence = prior_p;  // Map 0→0, 1→1

                    // Draw translucent blue overlay
                    cv::Mat roi = image(cv::Rect(x, y, cellSize, cellSize));
                    cv::Mat overlay(cellSize, cellSize, CV_8UC3, cv::Scalar(255, 0, 0)); // Blue
                    cv::addWeighted(overlay, confidence, roi, 1.0 - confidence, 0, roi);

                    // Draw actual prior_p value as text (not normalized)
                    std::string text = cv::format("%.2f", prior_p);
                    cv::putText(image, text, cv::Point(x + 25, y + 60),
                                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);

                } else {
                    cv::circle(image, center, 35, cv::Scalar(255, 0, 0), 4); // Blue circle
                }
        }
    
        cv::imwrite(filename, image);
        }
    }   

private:
    // Helper: Convert visit counts to action probabilities using temperature
    static std::vector<double> visit_count_to_action_distribution(const std::vector<double>& visits, double temperature) {
        if (temperature == 0) {
            throw std::invalid_argument("Temperature cannot be 0");
        }

        if (std::all_of(visits.begin(), visits.end(), [](double v){ return v == 0; })) {
            throw std::invalid_argument("All visit counts cannot be 0");
        }

        std::vector<double> normalized_visits(visits.size());

        for (size_t i = 0; i < visits.size(); i++) {
            normalized_visits[i] = visits[i] / temperature;
        }

        double sum = std::accumulate(normalized_visits.begin(), normalized_visits.end(), 0.0);

        for (double& visit : normalized_visits) {
            visit /= sum;
        }

        return normalized_visits;
    }

    // Helper: Softmax function to normalize values
    static std::vector<double> softmax(const std::vector<double>& values, double temperature) {
        std::vector<double> exps;
        double sum = 0.0;
        double max_value = *std::max_element(values.begin(), values.end());

        for (double v : values) {
            double exp_v = std::exp((v - max_value) / temperature);
            exps.push_back(exp_v);
            sum += exp_v;
        }

        for (double& exp_v : exps) {
            exp_v /= sum;
        }

        return exps;
    }

    // Helper: Randomly choose an action based on probabilities
    static int random_choice(const std::vector<int>& actions, const std::vector<double>& probs) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::discrete_distribution<> d(probs.begin(), probs.end());
        return actions[d(gen)];
    }
};

// Bind Node and MCTS to the same pybind11 module
PYBIND11_MODULE(mcts_alphazero, m) {
    // Bind the Node class
    py::class_<Node, std::shared_ptr<Node>>(m, "Node")
    .def(py::init([](std::shared_ptr<Node> parent){
        return std::make_shared<Node>(parent);
    }), py::arg("parent")=nullptr)
    .def("is_leaf", &Node::is_leaf)
    .def("is_root", &Node::is_root)
    .def_property_readonly("visit_count", &Node::get_visit_count);

    // Bind the MCTS class
    py::class_<MCTS>(m, "MCTS")
        .def(py::init<int, int, double, double, double, double, py::object>(),
             py::arg("max_moves")=512, py::arg("num_simulations")=800,
             py::arg("pb_c_base")=19652, py::arg("pb_c_init")=1.25,
             py::arg("root_dirichlet_alpha")=0.3, py::arg("root_noise_weight")=0.25, py::arg("simulate_env"))
        .def("_ucb_score", &MCTS::_ucb_score)
        .def("_add_exploration_noise", &MCTS::_add_exploration_noise)
        .def("_select_child", &MCTS::_select_child)
        .def("_expand_leaf_node", &MCTS::_expand_leaf_node)
        .def("get_next_action", &MCTS::get_next_action)
        .def("_simulateV", &MCTS::_simulateV)
        .def("_simulateQ", &MCTS::_simulateQ)
        .def_property("simulate_env", &MCTS::get_simulate_env, &MCTS::set_simulate_env)
        .def_property_readonly("pb_c_base", &MCTS::get_pb_c_base)
        .def_property_readonly("pb_c_init", &MCTS::get_pb_c_init)
        .def("get_next_action", &MCTS::get_next_action,
             py::arg("state_config_for_env_reset"),
             py::arg("policy_value_func"),
             py::arg("temperature"),
             py::arg("sample"));
}