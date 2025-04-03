#ifndef NODE_ALPHAZERO_H
#define NODE_ALPHAZERO_H

#include <map>
#include <string>
#include <memory>
#include <iostream>

class Node : public std::enable_shared_from_this<Node> {
public:
    // Parent and child nodes are managed using shared_ptr
    std::shared_ptr<Node> parent;
    std::map<int, std::shared_ptr<Node>> children;

    // Constructor
    Node(std::shared_ptr<Node> parent = nullptr, float prior_p = 1.0)
        : parent(parent), prior_p(prior_p), visit_count(0), value(0.0) {}

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

    float prior_p;        // The prior probability of the node
    int visit_count;      // Visit count
    float value;      // Value sum
};

#endif // NODE_ALPHAZERO_H