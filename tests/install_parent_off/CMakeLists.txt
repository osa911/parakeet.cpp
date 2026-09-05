cmake_minimum_required(VERSION 3.20)

project(parakeet_install_parent_off LANGUAGES CXX)

if(NOT DEFINED PARAKEET_SOURCE_DIR)
    message(FATAL_ERROR "PARAKEET_SOURCE_DIR is required")
endif()

set(PARAKEET_BUILD_CLI OFF)
set(PARAKEET_BUILD_TESTS OFF)
set(PARAKEET_BUILD_EXAMPLES OFF)
set(PARAKEET_INSTALL ON)
set(AXIOM_INSTALL OFF)

add_subdirectory("${PARAKEET_SOURCE_DIR}" parakeet)
