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
#include <string>
#include <fstream>
#include <cstdlib>  // for std::system
#include <chrono>
#include <sstream>

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
    double gamma;

public:
    // Constructor to initialize MCTS with optional parameters
    MCTS(int max_moves=512, int num_simulations=800,
         double pb_c_base=19652, double pb_c_init=1.25,
         double root_dirichlet_alpha=0.3, 
         double root_noise_weight=0.25, 
         py::object simulate_env=py::none(),
         double c=2.5,
         double p=3,
         double gamma=0.95)
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
    double _ucb_score(std::shared_ptr<Node> V_sh, std::shared_ptr<Node> Q_sh_a) {
        // Calculate PB-C component of UCB
        double pb_c = c * std::sqrt(std::sqrt(V_sh->visit_count)) / std::sqrt(Q_sh_a->visit_count);

        // Combine prior probability and value score
        double prior_score = pb_c * Q_sh_a->prior_p;
        double value_score = Q_sh_a->value;
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
    std::tuple<int, std::shared_ptr<Node>, std::shared_ptr<Node>> _select_child(std::shared_ptr<Node> V_sh, std::shared_ptr<Node> Q_sh, py::object simulate_env) {
        int action = -1;
        std::shared_ptr<Node> V_sh_plus_1 = nullptr;
        std::shared_ptr<Node> Q_sh_a = nullptr;
        double best_score = -9999999;

        // Iterate through all children
        for (const auto& kv : Q_sh->children) {
            int action_tmp = kv.first;
            std::shared_ptr<Node> Q_sh_a_tmp = kv.second;

            // Get legal actions from the simulation environment
            py::list legal_actions_py = simulate_env.attr("legal_actions").cast<py::list>();

            std::vector<int> legal_actions;
            for (py::handle h : legal_actions_py) {
                legal_actions.push_back(h.cast<int>());
            }

            // Check if the action is legal and calculate UCB score
            if (std::find(legal_actions.begin(), legal_actions.end(), action_tmp) != legal_actions.end()) {
                double score = _ucb_score(V_sh, Q_sh_a_tmp);
                if (score > best_score) {
                    best_score = score;
                    action = action_tmp;
                    V_sh_plus_1 = V_sh->children[action_tmp];
                    Q_sh_a = Q_sh->children[action_tmp];
                }
            }
        }
        // If no valid child is found, return the current node
        if ((Q_sh_a == nullptr) && (V_sh_plus_1 == nullptr)) {
            V_sh_plus_1 = V_sh;
            Q_sh_a = Q_sh;
        }

        return std::make_tuple(action, V_sh_plus_1, Q_sh_a);
    }

    // Expand a leaf node by adding its children based on policy probabilities
    double _expand_leaf_node(std::shared_ptr<Node> V_sh, std::shared_ptr<Node> Q_sh, py::object simulate_env, py::object policy_value_func) {
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
                V_sh->children[action] = std::make_shared<Node>(V_sh, prior_p);
                Q_sh->children[action] = std::make_shared<Node>(Q_sh, prior_p);
            }
        }
        return leaf_value;
    }

    // Main function to get the next action from MCTS
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

        // Expand the root node
        _expand_leaf_node(V_root, Q_root, simulate_env, policy_value_func);
        V_root->visit_count++;

        if (sample) {
            _add_exploration_noise(V_root);
            _add_exploration_noise(Q_root);
        }

        py::list legal_actions_py = simulate_env.attr("legal_actions").cast<py::list>();

        std::vector<int> legal_actions;
        for (py::handle h : legal_actions_py) {
            legal_actions.push_back(h.cast<int>());
        }
        // py::print("legal_actions:", legal_actions);

        // Run MCTS simulations
        for (int n = 1; n < num_simulations; ++n) {
            simulate_env.attr("reset")(
                state_config_for_env_reset["start_player_index"].cast<int>(),
                init_state,
                state_config_for_env_reset["katago_policy_init"].cast<bool>(),
                katago_game_state
            );
            simulate_env.attr("battle_mode") = simulate_env.attr("battle_mode_in_simulation_env");
            _simulateV(V_root, Q_root, simulate_env, policy_value_func);

            // Export the tree structure
            // exportToDot(-1, "/content/tree.dot", V_root, Q_root);
            // std::string command = "dot -Tpng -Gdpi=300 /content/tree.dot -o /content/log/tree_legal_ac_len_" + std::to_string(legal_actions.size()) + "_num_sti_" + std::to_string(n) + ".png";
            // int ret = std::system(command.c_str());        
        }

        // Collect visit counts from the root's children
        std::vector<std::pair<int, int>> action_visits;
        for (int action = 0; action < simulate_env.attr("action_space").attr("n").cast<int>(); ++action) {
            if (Q_root->children.count(action)) {
                action_visits.emplace_back(action, Q_root->children[action]->visit_count);
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
        return std::make_tuple(action_selected, action_probs, Q_root);
    }

    // Simulate a game starting from a given node
    double _simulateV(std::shared_ptr<Node> V_sh, std::shared_ptr<Node> Q_sh, py::object simulate_env, py::object policy_value_func) {
        int action;
        std::shared_ptr<Node> V_sh_plus_1;
        std::shared_ptr<Node> Q_sh_a;

        // Select action
        std::tie(action, V_sh_plus_1, Q_sh_a) = _select_child(V_sh, Q_sh, simulate_env);

        // StimulateQ
        double leaf_value = _simulateQ(action, V_sh_plus_1, Q_sh_a, simulate_env, policy_value_func);
        
        // Update V_sh
        V_sh->visit_count++;
        double tmp = 0.0;
        int total_visit = 0;

        // Get total visit count through all Q_node
        for (const auto& kv : Q_sh->children) {
            int action_tmp = kv.first;
            std::shared_ptr<Node> Q_sh_a_tmp = kv.second;
            for (const auto& kv1 : Q_sh_a_tmp->children) {
                int action_tmp1 = kv1.first;
                std::shared_ptr<Node> Q_sh_a_tmp1 = kv1.second;
                total_visit += Q_sh_a_tmp1->visit_count;
            }
        }  

        // Updating V_value
        for (const auto& kv : Q_sh->children) {
            int action_tmp = kv.first;
            std::shared_ptr<Node> Q_sh_a_tmp = kv.second;
            for (const auto& kv1 : Q_sh_a_tmp->children) {
                int action_tmp1 = kv1.first;
                std::shared_ptr<Node> Q_sh_a_tmp1 = kv1.second;

                tmp += ((double) Q_sh_a_tmp1->visit_count / total_visit) * pow(Q_sh_a_tmp1->value, p);
            }
        }

        // If there is no child node then not updating V_sh value 
        if (total_visit != 0) {
            V_sh->value = pow(tmp, 1.0 / p);
        }
        return leaf_value;
    }

    double _simulateQ(int action, std::shared_ptr<Node> V_sh_plus_1, std::shared_ptr<Node> Q_sh_a, py::object simulate_env, py::object policy_value_func) {
        // Apply action to env
        simulate_env.attr("step")(action);

        // Check if game is ended
        bool done;
        int winner;
        double leaf_value;

        py::tuple game_result = simulate_env.attr("get_done_winner")();
        done = game_result[0].cast<bool>();
        winner = game_result[1].cast<int>();

        if (!done) {
            if (V_sh_plus_1->is_leaf()) {
                leaf_value = _expand_leaf_node(V_sh_plus_1, Q_sh_a, simulate_env, policy_value_func);
                
                // Rescale leaf_value
                leaf_value = std::max(0.0, (leaf_value + 1) / 2);

                // Update V_sh_plus_1
                V_sh_plus_1->visit_count++;
                V_sh_plus_1->value = leaf_value; 
                V_sh_plus_1->flag = 3;
            } else {
                leaf_value = 1.0 - _simulateV(V_sh_plus_1, Q_sh_a, simulate_env, policy_value_func);

                V_sh_plus_1->flag = -1;
            }
        } else {
            if (winner == -1) {
                leaf_value = 0;

                V_sh_plus_1->flag = 0;
            } else {
                leaf_value = (simulate_env.attr("current_player").cast<int>() == winner) ? -1 : 1;
                
                if (leaf_value == -1) {
                    V_sh_plus_1->flag = 1;
                } else {
                    V_sh_plus_1->flag = 2;
                }
            }

            // Re-scale leaf_value
            leaf_value = std::max(0.0, (leaf_value + 1) / 2);

            // Update V_sh_plus_1
            V_sh_plus_1->visit_count++;
            V_sh_plus_1->value = leaf_value;     
        }
        Q_sh_a->value = (Q_sh_a->value * Q_sh_a->visit_count + leaf_value + gamma * V_sh_plus_1->value) / (Q_sh_a->visit_count + 1);

        Q_sh_a->visit_count = Q_sh_a->visit_count + 1;
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
            fprintf(file, "    %d [label=\"Ac: %d\\nvisit:%.0f\", fontsize=8, width=0.6, height=0.6, fixedsize=true, "
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

            if (V_sh_plus_1->is_leaf() && (V_sh_plus_1->visit_count == 0)) {
                continue;
            }

            int childId = nodeToId[V_sh_plus_1];

            double penwidth = 2.0 + Q_sh_a_tmp->value;
            
            // Ensure penwidth is a valid number
            if (std::isnan(penwidth) || std::isinf(penwidth) || penwidth <= 0.0) {
                penwidth = 2.0; // fallback to default thickness
            }

            int red = std::min(255, static_cast<int>(Q_sh_a_tmp->value * 20));
            if (std::isnan(red) || std::isinf(red)) {
                int red = 5;
            }

            int green = 255 - red;
            int blue = 255 - red;

            char colorStr[16];
            snprintf(colorStr, sizeof(colorStr), "#%02X%02X%02X", red, green, blue);
            
            // Write edge information
            fprintf(file, "    %d -> %d [label=\"Q_val: %.2f\\nprior_p:%.2f\", penwidth=%.2f, color=\"%s\", style=bold];\n",
                    nodeId, childId, Q_sh_a_tmp->value, Q_sh_a_tmp->prior_p, penwidth, colorStr);

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