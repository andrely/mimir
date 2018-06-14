cmake_minimum_required(VERSION 3.8)
project(CPP)

set(CMAKE_CXX_STANDARD 11)

include_directories("/usr/local/include/eigen3")

set(SOURCE_FILES src/main.cpp src/data.cpp src/data.h)
add_executable(CPP ${SOURCE_FILES})

project(logistic-test)
set(MIMIR_FILES src/data.cpp src/data.h)
add_executable(logistic-test ${MIMIR_FILES} src/examples/logistic-test.cpp)