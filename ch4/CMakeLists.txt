cmake_minimum_required(VERSION 3.0)
project(useSophus)

# 为使用 sophus，需要使用find_package命令找到它
find_package(Sophus REQUIRED)
find_package(fmt REQUIRED)

#set(CMAKE_CXX_FLAGS "-std=c++11")

# Eigen
include_directories("/usr/include/eigen3")
#include_directories("/usr/include/fmt")
add_executable(useSophus useSophus.cpp)
target_link_libraries(useSophus fmt::fmt)
#target_link_libraries(useSophus fmt::fmt-header-only)
target_link_libraries(useSophus Sophus::Sophus)

add_subdirectory(example)
