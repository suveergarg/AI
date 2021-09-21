cmake_minimum_required(VERSION 3.10)
project(POSE_GRAPH)
find_package(Ceres REQUIRED)

add_executable(ceres_hello_world ceres-hello-world.cpp)
target_link_libraries(ceres_hello_world Ceres::ceres)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
