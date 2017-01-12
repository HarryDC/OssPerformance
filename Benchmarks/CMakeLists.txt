# This file is a part of the SimQuest OpenSurgSim extension.
# Copyright 2012-2016, SimQuest Solutions Inc.

set(BENCHMARK_DIR "" CACHE PATH "Folder containing benchmark")

find_path(benchmark_INCLUDE_DIR "benchmark/benchmark.h"
  PATHS ${BENCHMARK_DIR}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH)
find_path(benchmark_INCLUDE_DIR "benchmark/benchmark.h")

find_library(benchmark_LIBRARY NAMES "benchmark"
  PATHS ${BENCHMARK_DIR}
  PATH_SUFFIXES lib lib64
  NO_DEFAULT_PATH)
find_library(benchmark_LIBRARY NAMES "benchmark")

include_directories(${benchmark_INCLUDE_DIR})

add_subdirectory(StdLibrary)
add_subdirectory(Collision)
add_subdirectory(Compute)