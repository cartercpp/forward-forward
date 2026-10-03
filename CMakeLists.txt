cmake_minimum_required(VERSION 4.3)
project(ForwardForward)

set(CMAKE_CXX_STANDARD 26)

add_executable(ForwardForward main.cpp
        math_vector.hpp
        matrix.hpp
        forward_forward.hpp
        forward_forward.cpp)
