cmake_minimum_required(VERSION 3.20)

project(parakeet_install_consumer LANGUAGES CXX)

find_package(Axiom CONFIG REQUIRED)
find_package(Parakeet CONFIG REQUIRED)

add_executable(parakeet_install_consumer main.cpp)
target_link_libraries(parakeet_install_consumer PRIVATE Axiom::axiom Parakeet::parakeet)
