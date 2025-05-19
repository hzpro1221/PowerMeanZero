#ifndef NODE_ALPHAZERO_H
#define NODE_ALPHAZERO_H

#include <map>
#include <string>
#include <memory>
#include <iostream>
#include <list>

class Node : public std::enable_shared_from_this<Node> {
public:
    // Shared pointer to the parent node
    std::shared_ptr<Node> parent;
    
    // Map of child nodes keyed by action indices
    std::map<int, std::shared_ptr<Node>> children;
    
    // List of opponent moves leading to this node
    std::list<int> opp_mov;
    
    // Prior probabilities for actions (keyed by action)
    std::map<int, double> prior_p;

    // Constructor with optional parent node and default prior probability
    Node(std::shared_ptr<Node> parent = nullptr, double prior_p = 1.0)
        : parent(parent), visit_count(0), value(0.0) {}

    // Default destructor
    ~Node() = default;

    // Returns true if the node has no children (i.e., it is a leaf node)
    bool is_leaf() const {
        return children.empty();
    }

    // Returns true if the node has no parent (i.e., it is the root node)
    bool is_root() const {
        return parent == nullptr;
    }

    // Returns the number of times this node has been visited
    int get_visit_count() const { return visit_count; }

    int visit_count;      // Number of visits to this node
    double value;         // Estimated value of the node
    int flag;             // Debug flag: -1 = node expanded and not terminal,
                          //             0 = loss,
                          //             1 = draw,
                          //             2 = win,
                          //             3 = expanded node
    double init_reward;   // Initial reward value for the node
};

#endif // NODE_ALPHAZERO_H
