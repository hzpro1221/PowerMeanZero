#include "node_alphazero.h"
#include <cmath>
#include <map>
#include <random>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>

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
    float c;
    float gamma;             
    float p;
    int player_turn;

    public:
    // Constructor to initialize MCTS with optional parameters
    MCTS(int max_moves=512, int num_simulations=800,
        double pb_c_base=19652, double pb_c_init=3,
        double root_dirichlet_alpha=0.3, double root_noise_weight=0.25, py::object simulate_env=py::none(),
        float gamma = 0.99, float p = 2, float c = 1.25)
        : max_moves(max_moves), num_simulations(num_simulations),
        pb_c_base(pb_c_base), pb_c_init(pb_c_init),
        root_dirichlet_alpha(root_dirichlet_alpha),
        root_noise_weight(root_noise_weight),
        gamma(gamma),
        p(p),
        c(c),
        simulate_env(simulate_env) {}

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

    // Calculate the Upper Confidence Bound (UCB) score for child nodes
    double _ucb_score(std::shared_ptr<Node> V_sh, std::shared_ptr<Node> Q_sh_a) {
        // py::print("\t\t\t------------------------------------_UCB_SCORE------------------------------------");
        // Calculate PB-C component of UCB
        double pb_c = std::log((V_sh->visit_count + pb_c_base + 1) / pb_c_base) + pb_c_init;
        pb_c *= std::sqrt(std::sqrt(V_sh->visit_count) / (Q_sh_a->visit_count));

        // py::print("\t\t\tV_sh->visit_count:", V_sh->visit_count, "\n");

        // py::print("\t\t\tQ_sh_a->visit_count:", Q_sh_a->visit_count);
        // py::print("\t\t\tQ_sh_a->prior_p:", Q_sh_a->prior_p);
        // py::print("\t\t\tQ_sh_a->value:", Q_sh_a->value);
        // py::print("\t\tprior_p:", Q_sh_a->prior_p, "; Q_sh_a->prior_p * c", Q_sh_a->prior_p * c);
        // Combine prior probability and value score
        double prior_score = Q_sh_a->prior_p * c * std::sqrt(std::sqrt(V_sh->visit_count) / (Q_sh_a->visit_count));
        double value_score = Q_sh_a->value;
        // py::print("\t\t\tUCB score:", prior_score + value_score);
        // py::print("\t\t\t----------------------------------------------------------------------------------");        
        return prior_score + value_score;
    }

    // Add Dirichlet noise to the root node for exploration
    void _add_exploration_noise(std::shared_ptr<Node> node) {
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
            node->children[actions[i]]->prior_p = node->children[actions[i]]->prior_p * (1 - frac) + noise[i] * frac;
        }
    }

    // Select the best child node based on UCB score
    std::tuple<int, std::shared_ptr<Node>, std::shared_ptr<Node>> _select_child(std::shared_ptr<Node> V_sh, std::shared_ptr<Node> Q_sh, py::object simulate_env, py::object policy_value_func) {
        // py::print("\t\t-----------------------------------------SELECT CHILD-----------------------------------------");
        int action = -1;
        std::shared_ptr<Node> V_sh_plus_1 = nullptr;
        std::shared_ptr<Node> Q_sh_a = nullptr;
        double best_score = -9999999;
        
        // Iterate through all children
        for (const auto& kv : Q_sh->children) {
            int action_tmp = kv.first;
            std::shared_ptr<Node> Q_sh_a_temp = kv.second;

            // Get legal actions from the simulation environment
            py::list legal_actions_py = simulate_env.attr("legal_actions").cast<py::list>();
            
            std::vector<int> legal_actions;
            for (py::handle h : legal_actions_py) {
                legal_actions.push_back(h.cast<int>());
            }
            
            // Check if the action is legal and calculate UCB score
            if (std::find(legal_actions.begin(), legal_actions.end(), action_tmp) != legal_actions.end()) {
                // py::print("\t\tCACULATE UCB SCORE FOR ACTION:", action_tmp);
                double score = _ucb_score(V_sh, Q_sh_a_temp);
                if (score > best_score) {
                    best_score = score;
                    action = action_tmp;
                    Q_sh_a = Q_sh_a_temp;
                    V_sh_plus_1 = V_sh->children[action_tmp];
                }
            }
        }

        py::list legal_actions_py = simulate_env.attr("legal_actions").cast<py::list>();
            
        std::vector<int> legal_actions;
        for (py::handle h : legal_actions_py) {
            legal_actions.push_back(h.cast<int>());
        }        
        // py::print("legal_actions:", legal_actions); 

        // If no valid child is found, then extend the first action in legal_action list and choose that action as the next action
        if (action == -1) {
            V_sh_plus_1 = V_sh;
            Q_sh_a = Q_sh;
        }
        
        // py::print("\t\taction selected:", action, "- UCB score:", best_score);       
        // py::print("\t\t----------------------------------------------------------------------------------------------");        
        return std::make_tuple(action, V_sh_plus_1, Q_sh_a);
    }

    // Expand a leaf node by adding its children based on policy probabilities
    double _expand_leaf_node(std::shared_ptr<Node> V_sh, std::shared_ptr<Node> Q_sh, py::object simulate_env, py::object policy_value_func) {
        // py::print("\t\t--------------------------------------EXPAND LEAF--------------------------------------");
        std::map<int, double> action_probs_dict;
        double leaf_value;

        // Call the policy-value function to get action probabilities and leaf value
        py::tuple result = policy_value_func(simulate_env);
        // py::print("\t\taction_probs_dict:", result[0].cast<std::map<int, double>>(), 1);
        action_probs_dict = result[0].cast<std::map<int, double>>();
        // action_probs_dict = softmax(result[0].cast<std::map<int, double>>(), 1);
        
        leaf_value = result[1].cast<double>();

        // Get the legal actions from the simulation environment
        py::list legal_actions_list = simulate_env.attr("legal_actions").cast<py::list>();
        // Temoral comment this line of code; For play_with_bot purpose
        std::vector<int> legal_actions = legal_actions_list.cast<std::vector<int>>();

        // py::print("\t\tExtend Node:", legal_actions);
        // Add child nodes for legal actions
        // py::print("\t\tAction_probs_dict:", action_probs_dict);
        for (const auto& kv : action_probs_dict) {
            int action = kv.first;
            double prior_p = kv.second;
            if (std::find(legal_actions.begin(), legal_actions.end(), action) != legal_actions.end()) {
                // py::print("\t\tExpand action:", action, ", prior_p:", prior_p);
                V_sh->children[action] = std::make_shared<Node>(V_sh, prior_p);
                Q_sh->children[action] = std::make_shared<Node>(Q_sh, prior_p);
            }
        }
        // py::print("\t\tLeaf value:", leaf_value);
        // py::print("\t\t---------------------------------------------------------------------------------");
        return leaf_value;
    }

    // Main function to get the next action from MCTS
    std::tuple<int, std::vector<double>, std::shared_ptr<Node>> get_next_action(py::object state_config_for_env_reset, py::object policy_value_func, double temperature, bool sample) {
        // Initialize the first tree - which is the main tree
        std::shared_ptr<Node> V_root = std::make_shared<Node>();
        std::shared_ptr<Node> Q_root = std::make_shared<Node>();

        // Initialize the second tree
        std::shared_ptr<Node> V_root_2 = std::make_shared<Node>();
        std::shared_ptr<Node> Q_root_2 = std::make_shared<Node>();

        // Configure initial environment state
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

        // Expand and update value for root node 
        // py::print("-----------------------------------INITIALIZE ROOT NODE-----------------------------------");
        double leaf_value = _expand_leaf_node(V_root, Q_root, simulate_env, policy_value_func);
        V_root->visit_count++;
        V_root->value = leaf_value;
        Q_root->value = leaf_value + gamma * V_root->value;  
        Q_root->visit_count++;
        // py::print("V_root->children:", V_root->children);
        // py::print("V_root->value:", V_root->value);
        // py::print("V_root->visit_count:", V_root->visit_count, "\n");        
        
        // py::print("Q_root->children:", Q_root->children);
        // py::print("Q_root->value:", Q_root->value);
        // py::print("Q_root->visit_count:", Q_root->visit_count, "\n");        
        leaf_value = _expand_leaf_node(V_root_2, Q_root_2, simulate_env, policy_value_func);
        V_root_2->visit_count++;
        V_root_2->value = leaf_value;
        Q_root_2->value = leaf_value + gamma * V_root_2->value;  
        Q_root_2->visit_count++;

        // py::print("V_root_2->children:", V_root_2->children);
        // py::print("V_root_2->value:", V_root_2->value);
        // py::print("V_root_2->visit_count:", V_root_2->visit_count, "\n");        
        
        // py::print("Q_root_2->children:", Q_root_2->children);
        // py::print("Q_root_2->value:", Q_root_2->value);
        // py::print("Q_root_2->visit_count:", Q_root_2->visit_count);                
        // py::print("-------------------------------------------------------------------------------------------");
        if (sample) {
            _add_exploration_noise(V_root);
            _add_exploration_noise(Q_root);

            _add_exploration_noise(V_root_2);
            _add_exploration_noise(Q_root_2);            
        }

        py::list legal_actions_py = simulate_env.attr("legal_actions").cast<py::list>();
            
        std::vector<int> legal_actions;
        for (py::handle h : legal_actions_py) {
            legal_actions.push_back(h.cast<int>());
        }
        
        // Run MCTS simulations
        for (int n = 1; n < num_simulations; ++n) {
            // py::print("-----------------------------------NUM STIMULATION:", n, "-----------------------------------"); 
            simulate_env.attr("reset")(
                state_config_for_env_reset["start_player_index"].cast<int>(),
                init_state,
                state_config_for_env_reset["katago_policy_init"].cast<bool>(),
                katago_game_state
            );
            simulate_env.attr("battle_mode") = simulate_env.attr("battle_mode_in_simulation_env");
            // py::print("battle_mode:", simulate_env.attr("battle_mode_in_simulation_env"));
            // Reset player_turn back to 1
            player_turn = 1;
            
            _simulateV(V_root, Q_root, V_root_2, Q_root_2, simulate_env, policy_value_func);
            // py::print("-------------------------------------------------------------------------------------------");
        }
        simulate_env.attr("reset")(
            state_config_for_env_reset["start_player_index"].cast<int>(),
            init_state,
            state_config_for_env_reset["katago_policy_init"].cast<bool>(),
            katago_game_state
        );
        simulate_env.attr("battle_mode") = simulate_env.attr("battle_mode_in_simulation_env");        
        // py::print("player:", simulate_env.attr("current_player").cast<int>());
        // py::print("legal_actions:", legal_actions);

        // Collect visit counts from the root's children
        std::vector<std::pair<int, int>> action_visits;
        for (int action = 0; action < simulate_env.attr("action_space").attr("n").cast<int>(); ++action) {
            if (Q_root->children.count(action)) {
                action_visits.emplace_back(action, Q_root->children[action]->visit_count);
                // py::print("Action:", action, "Visit Count:", Q_root->children[action]->visit_count, ", Q_value:", Q_root->children[action]->value, ", prior_p:", Q_root->children[action]->prior_p);
            }
            else {
                action_visits.emplace_back(action, 0);
                // py::print("Action:", action, "Visit Count:", 0);
            }
        }

        std::vector<int> actions;
        std::vector<int> visits;
        for (const auto& av : action_visits) {
            actions.emplace_back(av.first);
            visits.emplace_back(av.second);
        }

        std::vector<double> visits_d(visits.begin(), visits.end());
        std::vector<double> action_probs = visit_count_to_action_distribution(visits_d, temperature);

        int action_selected;
        if (sample) {
            action_selected = random_choice(actions, action_probs);
        }
        else {
            action_selected = actions[std::distance(action_probs.begin(), std::max_element(action_probs.begin(), action_probs.end()))];
        }
        // py::print("Action Selected:", action_selected, "Action Probabilities:", action_probs, "\n");
        // Return the selected action, action probabilities, and root node
        return std::make_tuple(action_selected, action_probs, Q_root);
    }

    // Simulate a game starting from a given node
    double _simulateV(std::shared_ptr<Node> V_sh, std::shared_ptr<Node> Q_sh, std::shared_ptr<Node> V_sh_2, std::shared_ptr<Node> Q_sh_2, py::object simulate_env, py::object policy_value_func) {
        // py::print("\t------------------------------------------------------STIMULATE V", simulate_env.attr("current_player").cast<int>(), "------------------------------------------------------");
        // Swap player turn
        double leaf_value;
        
        // Select next action
        int action;
        std::shared_ptr<Node> V_sh_plus_1 = nullptr;
        std::shared_ptr<Node> Q_sh_a = nullptr;
        
        std::tie(action, V_sh_plus_1, Q_sh_a) = _select_child(V_sh, Q_sh, simulate_env, policy_value_func);
        // py::print("\tAction selected:", action, "\n");

        // StimulateQ
        leaf_value = _simulateQ(action, V_sh_plus_1, Q_sh_a, V_sh_2, Q_sh_2, simulate_env, policy_value_func);
        
        // Update V node
        // py::print("\tUpdating V_node...");
        V_sh->visit_count++;
        V_sh->value = 0.0;
        for (const auto& kv : Q_sh->children) {
            int action_tmp = kv.first;
            std::shared_ptr<Node> Q_sh_a_temp = kv.second;
            
            // py::print("\tUPDATE - action:", action_tmp, ", (Q_sh_a_temp->visit_count / V_sh->visit_count) * pow(Q_sh_a_temp->value, p) =", (Q_sh_a_temp->visit_count / V_sh->visit_count) * pow(Q_sh_a_temp->value, p));
            // py::print("\t\tQ_sh_a_temp->visit_count:", Q_sh_a_temp->visit_count);
            // py::print("\t\tV_sh->visit_count:", V_sh->visit_count);
            // py::print("\t\tpow(Q_sh_a_temp->value, p):", pow(Q_sh_a_temp->value, p));
            
            V_sh->value += ((double) Q_sh_a_temp->visit_count / V_sh->visit_count) * pow(Q_sh_a_temp->value, p);
            // py::print("\t\tCurrent V_sh->value:", V_sh->value, "\n");
        }
        V_sh->value = pow(V_sh->value, 1 / p);
        // py::print("\tupdated V_sh->value:", V_sh->value);
        // py::print("\tupdated V_sh->visit_count:", V_sh->visit_count);
        // py::print("\t-------------------------------------------------------------------------------------------------------------------------");
        return leaf_value;
    }

    double _simulateQ(int action, std::shared_ptr<Node> V_sh_plus_1, std::shared_ptr<Node> Q_sh_a, std::shared_ptr<Node> V_sh_2, std::shared_ptr<Node> Q_sh_2, py::object simulate_env, py::object policy_value_func) {
        // py::print("\t------------------------------------------------------STIMULATE Q", simulate_env.attr("current_player").cast<int>(), "------------------------------------------------------");                
        double leaf_value;
        if (action != -1) {
            // Apply action to env
            simulate_env.attr("step")(action);

            // Continue to select
            leaf_value = _simulateV(V_sh_2, Q_sh_2, V_sh_plus_1, Q_sh_a, simulate_env, policy_value_func);
            
            // Swap the returned leaf_value
            leaf_value = 1.0 - leaf_value;
        } else if (action == -1) {
            // Check if game is finished and get the reward
            bool done;
            int winner;

            py::tuple game_result = simulate_env.attr("get_done_winner")();
            done = game_result[0].cast<bool>();
            winner = game_result[1].cast<int>();

            if (!done) {
                // Then extend node
                leaf_value = _expand_leaf_node(V_sh_plus_1, Q_sh_a, simulate_env, policy_value_func);

                // Re-scale leaf_value
                leaf_value = std::max(0.0, (leaf_value + 1.0) / 2.0);
                // leaf_value = leaf_value + 1.0;

                // Update node
                V_sh_plus_1->value = leaf_value;
                V_sh_plus_1->visit_count++; 
            } else {
                std::string battle_mode = simulate_env.attr("battle_mode_in_simulation_env").cast<std::string>();
                if (battle_mode == "self_play_mode") {
                    if (winner == -1) {
                        leaf_value = 0;
                    } else {
                        leaf_value = (simulate_env.attr("current_player").cast<int>() == winner) ? 1 : -1;
                    }
                }

                // Re-scale leaf_value
                leaf_value = std::max(0.0, (leaf_value + 1.0) / 2.0);
                // leaf_value = leaf_value + 1.0;

                // Update node
                V_sh_plus_1->value = leaf_value;
                V_sh_plus_1->visit_count++; 
            }
        }

        // Update Q node
        // py::print("Updating Q node...");

        Q_sh_a->value = (Q_sh_a->value * Q_sh_a->visit_count + leaf_value + gamma * V_sh_plus_1->value) / (Q_sh_a->visit_count + 1);
        Q_sh_a->visit_count++;
        // py::print("Updated Q_sh_a->value:", Q_sh_a->value);
        // py::print("Updated Q_sh_a->visit_count:", Q_sh_a->visit_count);
        
        // py::print("\t-------------------------------------------------------------------------------------------------------------------------");
        return leaf_value;
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
    static std::map<int, double> softmax(const std::map<int, double> action_probs_dict, double temperature) {
        std::map<int, double> exps;
        double sum = 0.0;

        // Find the maximum value in the action_probs_dict to avoid overflow
        double max_value = std::max_element(action_probs_dict.begin(), action_probs_dict.end(),
                                            [](const std::pair<int, double>& a, const std::pair<int, double>& b) {
                                                return a.second < b.second;
                                            })->second;
        // Calculate exponentials of the logits adjusted by temperature
        for (const auto& kv : action_probs_dict) {
            double exp_v = std::exp((kv.second - max_value) / temperature);
            exps[kv.first] = exp_v;  // Store in the map with the same action key
            sum += exp_v;
        }
    
        // Normalize the exponentials
        for (auto& kv : exps) {
            kv.second /= sum;
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
        .def(py::init([](std::shared_ptr<Node> parent, float prior_p){
            return std::make_shared<Node>(parent, prior_p);
        }), py::arg("parent")=nullptr, py::arg("prior_p")=1.0)
        .def("is_leaf", &Node::is_leaf)
        .def("is_root", &Node::is_root)
        .def("visit_count", [](const std::shared_ptr<Node>& self) {
            return self->visit_count;  
        });     

    // Bind the MCTS class
    py::class_<MCTS>(m, "MCTS")
        .def(py::init<int, int, double, double, double, double, py::object, float, float>(),
            py::arg("max_moves")=512, py::arg("num_simulations")=800,
            py::arg("pb_c_base")=19652, py::arg("pb_c_init")=1.25,
            py::arg("root_dirichlet_alpha")=0.3, py::arg("root_noise_weight")=0.25, py::arg("simulate_env"),
            py::arg("gamma")=0.99, py::arg("p")=1.5)
        .def("get_next_action", &MCTS::get_next_action)
        .def("_expand_leaf_node", &MCTS::_expand_leaf_node)
        .def_property("simulate_env", &MCTS::get_simulate_env, &MCTS::set_simulate_env)
        .def_property_readonly("pb_c_base", &MCTS::get_pb_c_base)
        .def_property_readonly("pb_c_init", &MCTS::get_pb_c_init)
        .def("get_next_action", &MCTS::get_next_action,
            py::arg("state_config_for_env_reset"),
            py::arg("policy_value_func"),
            py::arg("temperature"),
            py::arg("sample"));
}