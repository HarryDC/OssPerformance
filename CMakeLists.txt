# This file is a part of the SimQuest OpenSurgSim extension.
# Copyright 2012-2016, SimQuest Solutions Inc.

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/CMake)

add_subdirectory(Deformables)
add_subdirectory(Benchmarks)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/Data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})