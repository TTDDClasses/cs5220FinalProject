cmake_minimum_required(VERSION 3.11)
project(final_project_gpu)
enable_language(CUDA)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
add_compile_definitions(CUB_IGNORE_DEPRECATED_CPP_DIALECT=1)
add_compile_definitions(THRUST_IGNORE_DEPRECATED_CPP_DIALECT=1)

message(STATUS "Value of ALL_SIZES: ${ALL_SIZES}")

# We require the GNU compiler for this assignment.
if (NOT "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    set(Prg-Intel "PrgEnv-intel")
    set(Prg-Clang "PrgEnv-cray")
    message(WARNING
            "Must use GNU Compiler for submission. Make sure you ran:\n"
            "module swap ${Prg-${CMAKE_C_COMPILER_ID}} PrgEnv-gnu")
endif ()

add_executable(gpu main.cu gpu.cu ../common.h)
target_compile_features(gpu PRIVATE cxx_std_11)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_60")

# Copy the job scripts
configure_file(job-gpu job-gpu COPYONLY)
