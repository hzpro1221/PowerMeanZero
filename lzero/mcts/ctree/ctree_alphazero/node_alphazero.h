#ifndef NODE_ALPHAZERO_H
#define NODE_ALPHAZERO_H

#include <map>
#include <string>
#include <memory>
#include <iostream>
#include <vector>

class Node : public std::enable_shared_from_this<Node> {
public:
    // Parent and child nodes are managed using shared_ptr
    std::shared_ptr<Node> parent;
    std::map<int, std::shared_ptr<Node>> children;
    
    std::vector<int> opp_mov;
    std::map<int, double> prior_p; // The prior probability of the node

    // Constructor
    Node(std::shared_ptr<Node> parent = nullptr)
        : parent(parent), visit_count(0), value(0.0) {}

    // Default destructor
    ~Node() = default;

    // Check if the node is a leaf node
    bool is_leaf() const {
        return children.empty();
    }

    // Check if the node is the root node
    bool is_root() const {
        return parent == nullptr;
    }

    int visit_count;      // Visit count
    float value;      // Value sum
};

#endif // NODE_ALPHAZERO_H