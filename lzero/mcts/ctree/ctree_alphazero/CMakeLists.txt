cmake_minimum_required(VERSION 3.4...3.18)
project(mcts_alphazero VERSION 1.0)

# Python support
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# OpenCV support
find_package(OpenCV REQUIRED)

# pybind11 support
add_subdirectory(pybind11)

# Your module
pybind11_add_module(mcts_alphazero mcts_alphazero.cpp node_alphazero.cpp)

# Include paths
target_include_directories(mcts_alphazero PRIVATE
    ${Python3_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

# Link dependencies
target_link_libraries(mcts_alphazero PRIVATE
    pybind11::module
    ${OpenCV_LIBS}
)

# Set Python standard
set_target_properties(mcts_alphazero PROPERTIES PYTHON_STANDARD ${Python3_VERSION})
