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
    double root_noise_weight;       // Weight for exploration n oise added to root node
    double gamma;                   // Gamma
    double p;
    int i = 0;

    py::object simulate_env;        // Python object representing the simulation environment

    public:
    // Constructor to initialize MCTS with optional parameters
    MCTS(int max_moves=512, int num_simulations=800,
         double pb_c_base=19652, double pb_c_init=1,
         double root_dirichlet_alpha=0.3, double root_noise_weight=0.25, py::object simulate_env=py::none(),
         double gamma=0.99, double p=2)
        : max_moves(max_moves), num_simulations(num_simulations),
          pb_c_base(pb_c_base), pb_c_init(pb_c_init),
          root_dirichlet_alpha(root_dirichlet_alpha),
          root_noise_weight(root_noise_weight),
          gamma(gamma), p(p),
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
    double _ucb_score(std::shared_ptr<Node> V_parent, std::shared_ptr<Node> Q_child) {
        
        py::print("\n\t\t--------------------------Caculate UCB value--------------------------");
        // Calculate PB-C component of UCB
        py::print("\t\tV_parent->visit_count:", V_parent->visit_count);
        py::print("\t\tQ_child->prior_p:", Q_child->prior_p);
        py::print("\t\tQ_child->visit_count:", Q_child->visit_count);
        py::print("\t\tQ_child->value:", Q_child->value);
        
        double pb_c = std::log((V_parent->visit_count + pb_c_base + 1) / pb_c_base) + pb_c_init;
        // double pb_c = 1.5;
        pb_c *= std::sqrt(std::sqrt(V_parent->visit_count)) / std::sqrt(Q_child->visit_count);

        // Combine prior probability and value score
        double prior_score = pb_c * Q_child->prior_p;

        double value_score = Q_child->value;
        py::print("\t\tUCB_score:", prior_score + value_score);
        py::print("\t\t--------------------------END Caculate UCB value--------------------------\n");

        return prior_score + value_score;
    }

    // Add Dirichlet noise to the root node for exploration
    void _add_exploration_noise(std::shared_ptr<Node> V_node, std::shared_ptr<Node> Q_node) {
        std::vector<int> actions;
        // Collect all child actions of the root node
        for (const auto& kv : V_node->children) {
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
            V_node->children[actions[i]]->prior_p = V_node->children[actions[i]]->prior_p * (1 - frac) + noise[i] * frac;
            Q_node->children[actions[i]]->prior_p = Q_node->children[actions[i]]->prior_p * (1 - frac) + noise[i] * frac;
        }
    }

    // Select the best child node based on Power-Mean formulate
    std::tuple<int, std::shared_ptr<Node>, std::shared_ptr<Node>> _select_child(std::shared_ptr<Node> V_node, std::shared_ptr<Node> Q_node, py::object simulate_env) {
        int action = -1;

        std::shared_ptr<Node> V_child = nullptr;
        std::shared_ptr<Node> Q_child = nullptr;

        double best_score = -9999999;

        // Iterate through all children
        py::print("\n\t---------Select action---------");
        py::print("\tLegal actions:", py::str(simulate_env.attr("legal_actions").cast<py::list>()));
        
        for (const auto& kv : Q_node->children) {
            int action_tmp = kv.first;
            std::shared_ptr<Node> Q_child_tmp = kv.second;
            
            // Get legal actions from the simulation environment
            py::list legal_actions_py = simulate_env.attr("legal_actions").cast<py::list>();

            std::vector<int> legal_actions;
            for (py::handle h : legal_actions_py) {
                legal_actions.push_back(h.cast<int>());
            }
            
            // Check if the action is legal and calculate Power-mean score
            if (std::find(legal_actions.begin(), legal_actions.end(), action_tmp) != legal_actions.end()) {
                double score = _ucb_score(V_node, Q_child_tmp);
                py::print("action:", action_tmp, "q_value:", Q_child_tmp->value, "v_value of action:", V_node->children[action_tmp]->value);
                if (score > best_score) {
                    best_score = score;
                    action = action_tmp;
                    Q_child = Q_child_tmp;
                    V_child = V_node->children[action];
                }
            }
        }

        // If no valid child is found, return the current node
        if ((V_child == nullptr) || (Q_child == nullptr)) {
            py::print("\tNo valid child found!");    
            Q_child = Q_node;
            V_child = V_node;
        }

        py::print("\t----------------------------------\n");

        return std::make_tuple(action, V_child, Q_child);
    }

    // Expand a leaf node by adding its children based on policy probabilities
    double _expand_leaf_node(std::shared_ptr<Node> V_node, std::shared_ptr<Node> Q_node, py::object simulate_env, py::object policy_value_func) {
        py::print("\n\t---------------------------EXPAND_NODE---------------------------");
        std::map<int, double> action_probs_dict;
        double reward;
        py::print("\tBefore expand:");
        py::print("\t\tV_node_children:", V_node->get_children());
        py::print("\t\tQ_node_children:", Q_node->get_children().size());
        // Call the policy-value function to get action probabilities and leaf value
        py::tuple result = policy_value_func(simulate_env);
        action_probs_dict = result[0].cast<std::map<int, double>>();
        
        reward = result[1].cast<float>();
        reward = std::max(0.0, (reward + 1)/2); // Rescale reward
        
        py::print("\t\tReward from value_function is:", reward);

        // Get the legal actions from the simulation environment
        py::list legal_actions_list = simulate_env.attr("legal_actions").cast<py::list>();
        std::vector<int> legal_actions = legal_actions_list.cast<std::vector<int>>();

        // Add v_nodes and q_nodes respectively for legal actions
        for (const auto& kv : action_probs_dict) {
            int action = kv.first;
            double prior_p = kv.second;
            if (std::find(legal_actions.begin(), legal_actions.end(), action) != legal_actions.end()) {
                V_node->children[action] = std::make_shared<Node>(V_node, prior_p);
                Q_node->children[action] = std::make_shared<Node>(Q_node, prior_p);
            }
        }
        py::print("\n\tAfter expand:");
        py::print("\t\tV_node_children:", V_node->children);
        py::print("\t\tQ_node_children:", Q_node->children);        
        py::print("\t---------------------------END EXPAND_NODE---------------------------\n");
        return reward;
    }

    // Main function to get the next action from MCTS
    // This is the main loop of the Monte-Carlo Tree
    std::tuple<int, std::vector<double>, std::shared_ptr<Node>> get_next_action(py::object state_config_for_env_reset, py::object policy_value_func, double temperature, bool sample) {
        std::shared_ptr<Node> V_root = std::make_shared<Node>();
        std::shared_ptr<Node> Q_root = std::make_shared<Node>();

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
        simulate_env.attr("battle_mode") = simulate_env.attr("battle_mode_in_simulation_env");

        auto result = _simulateV(V_root, Q_root, simulate_env, policy_value_func);
        
        if (sample) {
            _add_exploration_noise(V_root, Q_root);
        }

        py::print("V_root - num_child: " + std::to_string(V_root->get_children().size()));
        py::print("Q_root - num_child: " + std::to_string(Q_root->get_children().size()));

        // Run MCTS simulations
        for (int n = 1; n < num_simulations; ++n) {
            py::print("NUM_STIMULATION: " + std::to_string(n));
            simulate_env.attr("reset")(
                state_config_for_env_reset["start_player_index"].cast<int>(),
                init_state,
                state_config_for_env_reset["katago_policy_init"].cast<bool>(),
                katago_game_state
            );
            simulate_env.attr("battle_mode") = simulate_env.attr("battle_mode_in_simulation_env");
            // This '_simulate' function equal 'StimulateV' as description in the paper :vv
            _simulateV(V_root, Q_root, simulate_env, policy_value_func);
        }

        // Collect visit counts from the root's children
        std::vector<std::pair<int, int>> action_visits;
        for (int action = 0; action < simulate_env.attr("action_space").attr("n").cast<int>(); ++action) {
            if (Q_root->children.count(action)) {
                py::print("is_child - action:", action, ", visit:", Q_root->children[action]->visit_count);
                action_visits.emplace_back(action, Q_root->children[action]->visit_count);
            }
            else {
                py::print("not_is_child - action:", action, ", visit:", 0);
                action_visits.emplace_back(action, 0);
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

        // Return the selected action, action probabilities, and root node
        return std::make_tuple(action_selected, action_probs, V_root);
    }

    // Stimulate Q-value
    double _stimulateQ(std::shared_ptr<Node> V_node_next, std::shared_ptr<Node> Q_node_next, int action, py::object simulate_env, py::object policy_value_func) {
        py::print("\n\t--------------------STIMULATE_Q--------------------");
        py::print("\tThe choosen V_node - visit count:", V_node_next->visit_count, ", V_value:", V_node_next->value);   
        py::print("\tThe choosen Q_node - visit count:", Q_node_next->visit_count, ", Q_value:", V_node_next->value);
        float V_next;
        
        // Apply action to environment to get to next state 
        simulate_env.attr("step")(action);

        // Get reward
        double reward;
        reward = std::get<2>(_check_game_result(simulate_env));

        if (reward == -1.0) {
            if (V_node_next->is_leaf()) {
                reward = _expand_leaf_node(V_node_next, Q_node_next, simulate_env, policy_value_func);
                V_node_next->value = reward; 
                V_node_next->visit_count++;

                V_next = V_node_next->value;
                py::print("\tGAME NOT END, NODE IS_LEAF - v_next:", V_next, ", reward", reward);
            } else {
                std::tie(reward, V_next) = _simulateV(V_node_next, Q_node_next, simulate_env, policy_value_func);
                py::print("\tGAME NOT END, NODE NOT IS_LEAF - v_next:", V_next, ", reward", reward);
            }
        }
        else {
            V_node_next->value = reward; 
            V_node_next->visit_count++;

            V_next = V_node_next->value;

            py::print("\tGAME END - V_next:", V_next, ", reward", reward);
        }

        py::print("\tQ_node_next->value:", Q_node_next->value);
        py::print("\tQ_node_next->visit_count:", Q_node_next->visit_count);        
        Q_node_next->value = (Q_node_next->value * Q_node_next->visit_count + reward + gamma * V_next) / (Q_node_next->visit_count + 1);
        Q_node_next->visit_count++;
        py::print("\tQ_value Updated:", Q_node_next->value);
        py::print("\tVisit Updated:", Q_node_next->visit_count);
        py::print("\t--------------------END STIMULATE_Q--------------------");
        return reward;
    }

    // Simulate a game starting from a given node
    std::pair<double, double> _simulateV(std::shared_ptr<Node> V_node, std::shared_ptr<Node> Q_node, py::object simulate_env, py::object policy_value_func) {
        py::print("\t--------------------STIMULATE_V--------------------");
        int action;
        double reward;

        std::shared_ptr<Node> V_child;
        std::shared_ptr<Node> Q_child;

        // Select next action
        std::tie(action, V_child, Q_child) = _select_child(V_node, Q_node, simulate_env);
        
        py::print("\t_stimulateV - action selected: " + std::to_string(action));
        py::print("\t_stimulateV - V_child->visit_count:", V_child->visit_count); 
        py::print("\t_stimulateV - Q_child->visit_count:", Q_child->visit_count);
        py::print("\t_stimulateV - V_child->is_leaf: " + std::to_string(V_child->is_leaf()));
        py::print("\t_stimulateV - Q_child->is_leaf: " + std::to_string(Q_child->is_leaf()));
        py::print("\t_stimulateV - V_child->children: ", V_child->children);
        py::print("\t_stimulateV - Q_child->children: ", Q_child->children);
        
        // If there is no action valid, then V_value as return value from roll-out policy
        if (action == -1) {
            reward = std::get<2>(_check_game_result(simulate_env));
            // std::cout << V_node.get() << " " << Q_node.get() << std::endl;
            if (reward == -1.0) {
                if (V_node->is_leaf()) {
                    reward = _expand_leaf_node(V_node, Q_node, simulate_env, policy_value_func);
                    V_node->value = reward;
                    py::print("\t_stimulateV - GAME IS NOT END - updated v_value:", reward);
                    V_node->visit_count++;
                    py::print("\t--------------------END STIMULATE_V--------------------");                    
                    return std::make_pair(reward, V_node->value);
                }
            } else {
                V_node->value = reward;
                py::print("\t_stimulateV - GAME IS END - updated v_value:", reward);
                V_node->visit_count++;
                py::print("\t--------------------END STIMULATE_V--------------------");                    
                return std::make_pair(reward, V_node->value);
            }
        } else {
            // Else run _stimulateQ
            reward = _stimulateQ(V_child, Q_child, action, simulate_env, policy_value_func);
        }

        // Get Q-value of children to caculate new V_value
        V_node->visit_count++;
        V_node->value = 0;
        for (const auto& kv : Q_node->children) {
            int action_tmp = kv.first;
            std::shared_ptr<Node> Q_child_tmp = kv.second;

            V_node->value += pow(Q_child_tmp->value, p) * ((double) Q_child_tmp->visit_count / V_node->visit_count);
            
            py::print("\n\t------------------------------------------------------------------------");
            py::print("\tElement of V_next caculation:");
            py::print("\tFor action", action_tmp, "- child q_value", Q_child_tmp->value ,"- child visit:", Q_child_tmp->visit_count, "- parent visit:", V_node->visit_count);
            py::print("\tpow(Q_child_tmp->value, p):", pow(Q_child_tmp->value, p));
            py::print("\t((double) Q_child_tmp->visit_count / V_node->visit_count):", ((double) Q_child_tmp->visit_count / V_node->visit_count));
            py::print("\t------------------------------------------------------------------------\n");
        }

        V_node->value = pow(V_node->value, 1/p);
        py::print("\t_stimulateV - updated v_value:", V_node->value);
        // Swap score for h - 1 depth if this game is "sel_play_mode"
        std::string battle_mode = simulate_env.attr("battle_mode_in_simulation_env").cast<std::string>();
        if (battle_mode == "self_play_mode") {
            reward =  std::max(0.0, (-(reward*2 - 1) + 1)/2); // Rescale reward
        } else if (battle_mode == "play_with_bot_mode") {
            reward = reward;
        }
        py::print("\t--------------------END STIMULATE_V--------------------");                    
        return std::make_pair(reward, V_node->value);
    }

    std::tuple<bool, int, double> _check_game_result(py::object simulate_env) {
        double reward = -1.0;
        
        bool done;
        int winner;
        py::tuple result = simulate_env.attr("get_done_winner")();
        done = result[0].cast<bool>();
        winner = result[1].cast<int>();        

        if (done) {
            std::string battle_mode = simulate_env.attr("battle_mode_in_simulation_env").cast<std::string>();
            if (battle_mode == "self_play_mode") {
                if (winner == -1) {
                    reward = 0;
                    reward = std::max(0.0, (reward + 1)/2); // Rescale
                } else {
                    reward = (simulate_env.attr("current_player").cast<int>() == winner) ? 1 : -1; 
                    reward = std::max(0.0, (reward + 1)/2); // Rescale                
                }
            }
            else if (battle_mode == "play_with_bot_mode") {
                if (winner == -1) {
                    reward = 0;
                    reward = std::max(0.0, (reward + 1)/2); // Rescale                    
                }
                else if (winner == 1) {
                    reward = 1;
                    reward = std::max(0.0, (reward + 1)/2); // Rescale                    
                }
                else if (winner == 2) {
                    reward = -1;
                    reward = std::max(0.0, (reward + 1)/2); // Rescale                    
                }
            }            
        }
        return std::make_tuple(done, winner, reward);
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
        .def(py::init<std::shared_ptr<Node>, float>(),
             py::arg("parent")=nullptr, py::arg("prior_p")=1.0)
        .def("is_leaf", &Node::is_leaf)
        .def("is_root", &Node::is_root)
        .def_property_readonly("parent", &Node::get_parent)
        .def_property_readonly("children", &Node::get_children)
        .def("add_child", &Node::add_child)
        .def_property_readonly("visit_count", [](const Node &n) { return n.visit_count; })
        .def_property_readonly("v_value", [](const Node &n) { return n.value; })
        .def_readwrite("prior_p", &Node::prior_p);

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
        .def_property("simulate_env", &MCTS::get_simulate_env, &MCTS::set_simulate_env)
        .def_property_readonly("pb_c_base", &MCTS::get_pb_c_base)
        .def_property_readonly("pb_c_init", &MCTS::get_pb_c_init)
        .def("get_next_action", &MCTS::get_next_action,
             py::arg("state_config_for_env_reset"),
             py::arg("policy_value_func"),
             py::arg("temperature"),
             py::arg("sample"));
}