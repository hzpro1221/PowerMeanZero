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
}