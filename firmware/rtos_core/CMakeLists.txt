cmake_minimum_required(VERSION 3.16)
project(ofnav_rtos_core LANGUAGES CXX)

option(OFNAV_BUILD_TESTS "Build RTOS core tests" ON)

add_library(ofnav_rtos_core STATIC
    src/ofnav.cpp
)

target_include_directories(ofnav_rtos_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(ofnav_rtos_core PUBLIC cxx_std_17)

target_compile_options(ofnav_rtos_core PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Werror
)

if(OFNAV_BUILD_TESTS)
    enable_testing()
    add_executable(test_ofnav tests/test_ofnav.cpp)
    target_link_libraries(test_ofnav PRIVATE ofnav_rtos_core)
    target_compile_options(test_ofnav PRIVATE -Wall -Wextra -Wpedantic -Werror)
    add_test(NAME test_ofnav COMMAND test_ofnav)
endif()
