cmake_minimum_required(VERSION 3.10)
project(POSE_GRAPH)

find_package(Ceres REQUIRED)
find_package(Boost REQUIRED)

add_executable(poseGraph2d poseGraph2d.cpp)
target_link_libraries(poseGraph2d Ceres::ceres)
target_link_libraries(poseGraph2d Boost::boost)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)