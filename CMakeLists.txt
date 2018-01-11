cmake_minimum_required(VERSION 3.9)
project(Retinex)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)

set(SOURCE_FILES main.cpp Retinex.hpp)
add_executable(Retinex ${SOURCE_FILES})
target_link_libraries(Retinex ${OpenCV_LIBS})