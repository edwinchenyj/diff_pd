cmake_minimum_required(VERSION 3.10)

project(diff_pd)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Pardiso requires stack variables to be aligned.
# Thanks to Fish for figuring out the flag -mpreferred-stack-boundary=4
# This magic number is suitable only for libpardiso600-GNU720-X86-64.so
add_definitions("-Wall -mpreferred-stack-boundary=4")
set(GCC_COVERAGE_COMPILE_FLAGS "-Wl,--no-as-needed")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")

# OpenMP.
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    add_definitions(-DEIGEN_DONT_PARALLELIZE)
endif()

# Pardiso.
option(PARDISO_AVAILABLE "Pardiso" ON)
if(PARDISO_AVAILABLE)
    add_definitions(-DPARDISO_AVAILABLE)
    add_library(pardiso SHARED IMPORTED)
    set_target_properties(pardiso PROPERTIES
        IMPORTED_LOCATION "$ENV{PARDISO_LIC_PATH}/libpardiso600-GNU720-X86-64.so"
    )
else()
    message("Pardiso not detected.")
endif()

# Include directories.
include_directories("../external/eigen")
include_directories("../external/spectra/include")
include_directories("./core/include")
execute_process(COMMAND bash -c "python3-config --includes|sed 's/-I//g'|awk '{print $1;}'" OUTPUT_VARIABLE PYTHON_INCLUDES)
include_directories(${PYTHON_INCLUDES})

# The core library.
file(GLOB_RECURSE CORE_CPP "./core/src/**/*.cpp")
add_library(diff_pd_core SHARED ${CORE_CPP})
if(PARDISO_AVAILABLE)
    target_link_libraries(diff_pd_core pardiso lapack blas OpenMP::OpenMP_CXX pthread m)
else()
    target_link_libraries(diff_pd_core OpenMP::OpenMP_CXX)
endif()

# Python wrapper library.
file(GLOB_RECURSE PYTHON_BINDING "./core/src/**/*.cpp" "./core/src/py_diff_pd_core_wrap.cxx")
add_library(py_diff_pd_core SHARED ${CORE_HEADER} ${PYTHON_BINDING})
if(PARDISO_AVAILABLE)
    target_link_libraries(py_diff_pd_core pardiso lapack blas OpenMP::OpenMP_CXX pthread m)
else()
    target_link_libraries(py_diff_pd_core OpenMP::OpenMP_CXX)
endif()

# Demo.
file(GLOB DEMO_CPP "./main.cpp")
add_executable(diff_pd_demo ${DEMO_CPP})
target_link_libraries(diff_pd_demo diff_pd_core)

include(FetchContent)
include(CTest)

set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

# Get Catch2
FetchContent_Declare(Catch2 URL
https://github.com/catchorg/Catch2/archive/v2.13.7.zip)
  # Use FetchContent to to download and make Catch2 part of our CMake build.
set(CATCH_INSTALL_HELPERS OFF)
set(CATCH_BUILD_EXAMPLES OFF)
set(CATCH_BUILD_EXTRA_TESTS OFF)
set(CATCH_BUILD_TESTING OFF)
set(CATCH_INSTALL_DOCS OFF)

FetchContent_MakeAvailable(Catch2)

file(GLOB test-sources CONFIGURE_DEPENDS tests/main.cxx)
add_executable(${PROJECT_NAME}-tests "${test-sources}")

# Mark the tests as needing Catch benchmark infrastructure which
# off by default
target_compile_definitions(${PROJECT_NAME}-tests
  PRIVATE
  CATCH_CONFIG_ENABLE_BENCHMARKING
  )

#
target_link_libraries(${PROJECT_NAME}-tests
  PRIVATE
    Catch2::Catch2
    diff_pd_core
  )

# catch_discover_tests will automatically convert each catch2
# test case into a separate ctest entry
include(${Catch2_SOURCE_DIR}/contrib/Catch.cmake)
catch_discover_tests(${PROJECT_NAME}-tests)


add_executable(${PROJECT_NAME}-siere-example tests/siere-example.cxx)

#
target_link_libraries(${PROJECT_NAME}-siere-example
  PRIVATE
    diff_pd_core
  )

  

add_executable(${PROJECT_NAME}-sibe-full-example tests/sibe-full-example.cxx)

#
target_link_libraries(${PROJECT_NAME}-sibe-full-example
  PRIVATE
    diff_pd_core
  )
