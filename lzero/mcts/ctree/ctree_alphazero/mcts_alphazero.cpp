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
    double gamma;                   // Gamma
    double p;
    int i = 0;

    py::object simulate_env;        // Python object representing the simulation environment

    public:
    // Constructor to initialize MCTS with optional parameters
    MCTS(int max_moves=512, int num_simulations=800,
         double pb_c_base=19652, double pb_c_init=1.25,
         double root_dirichlet_alpha=0.3, double root_noise_weight=0.25, py::object simulate_env=py::none(),
         double gamma=0.5, double p=1.5)
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
    double _ucb_score(std::shared_ptr<Node> parent, std::shared_ptr<Node> child) {
        
        py::print("\n\t\t--------------------------Caculate UCB value--------------------------");
        // Calculate PB-C component of UCB
        py::print("\t\tparent->visit_count:", parent->visit_count);
        py::print("\t\tchild->visit_count:", child->visit_count);
        py::print("\t\tchild->prior_p:", child->prior_p);
        py::print("\t\tchild->q_value:", child->q_value);
        double pb_c = std::log((parent->visit_count + pb_c_base + 1) / pb_c_base) + pb_c_init;
        pb_c *= std::sqrt(std::sqrt(parent->visit_count)) / std::sqrt(child->visit_count);

        // Combine prior probability and value score
        double prior_score = pb_c * child->prior_p;
        double value_score = child->q_value;
        py::print("\t\t--------------------------Caculate UCB value--------------------------\n");

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

    // Select the best child node based on Power-Mean formulate
    std::pair<int, std::shared_ptr<Node>> _select_child(std::shared_ptr<Node> node, py::object simulate_env) {
        int action = -1;
        std::shared_ptr<Node> child = nullptr;
        double best_score = -9999999;

        // Iterate through all children
        py::print("\n\t---------Select action---------");
        py::print("\tLegal actions:", py::str(simulate_env.attr("legal_actions").cast<py::list>()));
        
        for (const auto& kv : node->children) {
            int action_tmp = kv.first;
            std::shared_ptr<Node> child_tmp = kv.second;
            
            // Get legal actions from the simulation environment
            py::list legal_actions_py = simulate_env.attr("legal_actions").cast<py::list>();

            std::vector<int> legal_actions;
            for (py::handle h : legal_actions_py) {
                legal_actions.push_back(h.cast<int>());
            }
            // Check if the action is legal and calculate Power-mean score
            if (std::find(legal_actions.begin(), legal_actions.end(), action_tmp) != legal_actions.end()) {
                double score = _ucb_score(node, child_tmp);
                py::print("\tChild action:", action_tmp, " - v_value:", child_tmp->v_value," - q_value:", child_tmp->q_value,"- power-mean score:", score);
                if (score > best_score) {
                    best_score = score;
                    action = action_tmp;
                    child = child_tmp;
                }
            }
        }
        // If no valid child is found, return the current node
        if (child == nullptr) {
            child = node;
        }
        py::print("\t----------------------------------\n");

        return std::make_pair(action, child);
    }

    // Expand a leaf node by adding its children based on policy probabilities
    double _expand_leaf_node(std::shared_ptr<Node> node, py::object simulate_env, py::object policy_value_func) {
        std::map<int, double> action_probs_dict;
        double leaf_value;

        // Call the policy-value function to get action probabilities and leaf value
        py::tuple result = policy_value_func(simulate_env);
        action_probs_dict = result[0].cast<std::map<int, double>>();
        leaf_value = result[1].cast<double>();

        // Get the legal actions from the simulation environment
        py::list legal_actions_list = simulate_env.attr("legal_actions").cast<py::list>();
        std::vector<int> legal_actions = legal_actions_list.cast<std::vector<int>>();

        // Add child nodes for legal actions
        for (const auto& kv : action_probs_dict) {
            int action = kv.first;
            double prior_p = kv.second;
            if (std::find(legal_actions.begin(), legal_actions.end(), action) != legal_actions.end()) {
                node->children[action] = std::make_shared<Node>(node, prior_p);
            }
        }

        return leaf_value;
    }

    // Main function to get the next action from MCTS
    // This is the main loop of the Monte-Carlo Tree
    std::tuple<int, std::vector<double>, std::shared_ptr<Node>> get_next_action(py::object state_config_for_env_reset, py::object policy_value_func, double temperature, bool sample) {
        // Initialize root node
        std::shared_ptr<Node> root = std::make_shared<Node>();

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
        


        // Expand the root node
        // _expand_leaf_node(root, simulate_env, policy_value_func);
        _simulate(root, simulate_env, policy_value_func);
        if (sample) {
            _add_exploration_noise(root);
        }

        py::print("root - num_child: " + std::to_string(root->get_children().size()));
        for (const auto& [key, child] : root->get_children()) {
            py::print("action: " + std::to_string(key) + " - child->visit_count: " + std::to_string(child->visit_count));
        }

        // Run MCTS simulations
        for (int n = 0; n < num_simulations; ++n) {
            py::print("num_simulation: " + std::to_string(n));
            simulate_env.attr("reset")(
                state_config_for_env_reset["start_player_index"].cast<int>(),
                init_state,
                state_config_for_env_reset["katago_policy_init"].cast<bool>(),
                katago_game_state
            );
            simulate_env.attr("battle_mode") = simulate_env.attr("battle_mode_in_simulation_env");
            // This '_simulate' function equal 'StimulateV' as description in the paper :vv
            _simulate(root, simulate_env, policy_value_func);
        }

        // Collect visit counts from the root's children
        std::vector<std::pair<int, int>> action_visits;
        for (int action = 0; action < simulate_env.attr("action_space").attr("n").cast<int>(); ++action) {
            if (root->children.count(action)) {
                action_visits.emplace_back(action, root->children[action]->visit_count);
            }
            else {
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
        return std::make_tuple(action_selected, action_probs, root);
    }

    // Stimulate Q-value
    float _stimulateQ(std::shared_ptr<Node> node, int action, py::object simulate_env, py::object policy_value_func) {
        float v_next;
        
        // Apply action to environment to get to next state 
        simulate_env.attr("step")(action);        

        // Get reward
        double leaf_value;
        leaf_value = std::get<2>(_check_game_result(simulate_env));

        if (leaf_value == -1.0) {
            if (node->is_leaf()) {
                leaf_value = _expand_leaf_node(node, simulate_env, policy_value_func);
                v_next = leaf_value;
            } else {
                std::tie(leaf_value, v_next) = _simulate(node, simulate_env, policy_value_func);
            }
        }
        else {
            // What if s_(h+1) is terminal state? Then what is the V_(h+1) value?
            v_next = leaf_value;
        }

        node->q_value = (node->q_value + leaf_value + gamma * v_next) / (node->visit_count + 1);
        node->visit_count++;

        return leaf_value;
    }

    // Simulate a game starting from a given node
    std::pair<float, float> _simulate(std::shared_ptr<Node> node, py::object simulate_env, py::object policy_value_func) {
        int action;
        float leaf_value;
    
        std::shared_ptr<Node> child;

        // Select next action
        std::tie(action, child) = _select_child(node, simulate_env);
        py::print("\t_stimulate - action selected: " + std::to_string(action));
        py::print("\t_stimulate - child_selected.is_leaf ?: " + std::to_string(child->is_leaf()));
        // If there is no action valid, then V_value as return value from roll-out policy
        if (action == -1) {
            leaf_value = std::get<2>(_check_game_result(simulate_env));

            if (leaf_value == -1.0) {
                if (node->is_leaf()) {
                    leaf_value = _expand_leaf_node(node, simulate_env, policy_value_func);
                    node->v_value = leaf_value;
                    node->visit_count++;
                    return std::make_pair(leaf_value, node->v_value);
                }
            } else {
                node->v_value = leaf_value;
                node->visit_count++;
                return std::make_pair(leaf_value, node->v_value);
            }
        } else {
            // Else run _stimulateQ
            leaf_value = _stimulateQ(child, action, simulate_env, policy_value_func);
        }

        // Get Q-value of children to caculate new V_value
        double v_new = 0;
        for (const auto& kv : node->children) {
            int action_tmp = kv.first;
            std::shared_ptr<Node> child_tmp = kv.second;
            
            v_new += pow(child_tmp->q_value, p) * (child_tmp->visit_count / node->visit_count);
        }
        node->visit_count++;
        node->v_value = pow(v_new, 1/p);

        // Swap score for h - 1 depth if this game is "sel_play_mode"
        std::string battle_mode = simulate_env.attr("battle_mode_in_simulation_env").cast<std::string>();
        if (battle_mode == "self_play_mode") {
            leaf_value = -leaf_value;
        } else if (battle_mode == "play_with_bot_mode") {
            leaf_value = leaf_value;
        }
        return std::make_pair(leaf_value, node->v_value);
    }

    std::tuple<bool, int, float> _check_game_result(py::object simulate_env) {
        float leaf_value = -1.0;
        
        bool done;
        int winner;
        py::tuple result = simulate_env.attr("get_done_winner")();
        done = result[0].cast<bool>();
        winner = result[1].cast<int>();        

        if (done) {
            std::string battle_mode = simulate_env.attr("battle_mode_in_simulation_env").cast<std::string>();
            if (battle_mode == "self_play_mode") {
                if (winner == -1) {
                    leaf_value = 0;
                } else {
                    leaf_value = (simulate_env.attr("current_player").cast<int>() == winner) ? 1 : -1;
                }
            }
            else if (battle_mode == "play_with_bot_mode") {
                if (winner == -1) {
                    leaf_value = 0;
                }
                else if (winner == 1) {
                    leaf_value = 1;
                }
                else if (winner == 2) {
                    leaf_value = -1;
                }
            }            
        }
        return std::make_tuple(done, winner, leaf_value);
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
        .def_property_readonly("v_value", [](const Node &n) { return n.v_value; })
        .def_property_readonly("q_value", [](const Node &n) { return n.q_value; })
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
        .def("_simulate", &MCTS::_simulate)
        .def_property("simulate_env", &MCTS::get_simulate_env, &MCTS::set_simulate_env)
        .def_property_readonly("pb_c_base", &MCTS::get_pb_c_base)
        .def_property_readonly("pb_c_init", &MCTS::get_pb_c_init)
        .def("get_next_action", &MCTS::get_next_action,
             py::arg("state_config_for_env_reset"),
             py::arg("policy_value_func"),
             py::arg("temperature"),
             py::arg("sample"));
}