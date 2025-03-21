#include "node_alphazero.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(node_alphazero, m) {
    py::class_<Node, std::shared_ptr<Node>>(m, "Node")
        .def(py::init([](std::shared_ptr<Node> parent, float prior_p){
            return std::make_shared<Node>(parent, prior_p);
        }), py::arg("parent")=nullptr, py::arg("prior_p")=1.0)
        .def("is_leaf", &Node::is_leaf)
        .def("is_root", &Node::is_root)
        .def("parent", [](const Node& self) -> std::shared_ptr<Node> {
            return self.parent;
        })
        .def("children", [](const Node& self) -> std::map<int, std::shared_ptr<Node>> {
            return self.children;
        })
        .def("add_child", &Node::add_child)
        .def_property_readonly("visit_count", [](const Node &n) { return n.visit_count; })
        .def_readwrite("prior_p", &Node::prior_p)
        .def("get_child", [](const Node &self, int action) -> std::shared_ptr<Node> {
        auto it = self.children.find(action);
        if (it == self.children.end()) {
            return nullptr;
        }
        return it->second;
        })
        .def("get_q_nodes", [](const Node &self, int action) -> std::shared_ptr<Node> {
        auto it = self.q_nodes.find(action);
        if (it == self.q_nodes.end()) {
            return nullptr;
        }
        return it->second;
    });
}