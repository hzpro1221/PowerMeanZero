#ifndef NODE_ALPHAZERO_H
#define NODE_ALPHAZERO_H

#include <map>
#include <string>
#include <memory>
#include <iostream>
#include <list>

class Node : public std::enable_shared_from_this<Node> {
public:
    // Parent and child nodes are managed using shared_ptr
    std::shared_ptr<Node> parent;
    std::map<int, std::shared_ptr<Node>> children;
    
    std::list<int> opp_mov;
    std::map<int, double> prior_p;

    // Constructor
    Node(std::shared_ptr<Node> parent = nullptr, double prior_p = 1.0)
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

    // Get the visit count
    int get_visit_count() const { return visit_count; }

    int visit_count;      // Visit count
    double value;      // Value
    int flag;          // For debugging with 0 for lose, 1 for draw, 2 for win, 3 if extended
};

#endif // NODE_ALPHAZERO_H