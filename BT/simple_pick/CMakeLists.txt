cmake_minimum_required(VERSION 3.10.2)

project(bt_simple_pick)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_CLANG_TIDY "clang-tidy")

find_package(behaviortree_cpp)

add_executable(${PROJECT_NAME} "bt_simple_pick.cpp")
target_link_libraries(${PROJECT_NAME} BT::behaviortree_cpp)