cmake_minimum_required(VERSION 3.10)
project(POSE_GRAPH)

find_package(Ceres REQUIRED)
find_package(Boost REQUIRED)

add_executable(poseGraph3d poseGraph3d.cpp)
target_link_libraries(poseGraph3d Ceres::ceres)
target_link_libraries(poseGraph3d Boost::boost)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
