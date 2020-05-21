cmake_minimum_required(VERSION 2.8)

project(xor)
set(CMAKE_CXX_STANDARD 17)
add_executable(${PROJECT_NAME} "main.cpp")
target_compile_options(${PROJECT_NAME} PRIVATE -mavx2 -funroll-loops)
