cmake_minimum_required(VERSION 3.14)
project(final_project_cpu)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

# On Perlmutter -- 3.5 GHz * 4 vector width * 2 vector pipelines * 2 flops for FMA = 56 GF/s
set(MAX_SPEED 56 CACHE STRING "The max speed of the CPU in GF/s")

# We require the GNU compiler for this assignment.
if (NOT "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    set(Prg-Intel "PrgEnv-intel")
    set(Prg-Clang "PrgEnv-cray")
    message(WARNING
            "Must use GNU Compiler for submission. Make sure you ran:\n"
            "module swap ${Prg-${CMAKE_C_COMPILER_ID}} PrgEnv-gnu")
endif ()

add_executable(blas main.cpp blas.cpp)
add_executable(sparse main.cpp csr_csr.cpp)
# we need another one for GPU

# Copy the job scripts
# configure_file(job-serial job-serial COPYONLY)
# configure_file(job-openmp job-openmp COPYONLY)
