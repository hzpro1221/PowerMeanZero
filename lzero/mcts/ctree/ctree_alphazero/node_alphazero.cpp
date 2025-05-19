#include "node_alphazero.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(node_alphazero, m) {
    py::class_<Node, std::shared_ptr<Node>>(m, "Node")
        .def(py::init([](std::shared_ptr<Node> parent){
            return std::make_shared<Node>(parent);
        }), py::arg("parent")=nullptr)
        .def("is_leaf", &Node::is_leaf)
        .def("is_root", &Node::is_root)
        .def_property_readonly("visit_count", &Node::get_visit_count);
}
