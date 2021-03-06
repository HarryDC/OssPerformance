# This file is a part of the SimQuest OpenSurgSim extension.
# Copyright 2012-2016, SimQuest Solutions Inc.

link_directories(
	${Boost_LIBRARY_DIRS}
)

include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}
	"${Boost_INCLUDE_DIR}"
	"${EIGEN3_INCLUDE_DIR}"
)

set(BENCHMARK_SOURCES
	../Vectorize.cpp
	../Solvers.cpp
)

set(BENCHMARK_HEADERS

)

surgsim_add_executable(VectorizeAVX "${BENCHMARK_SOURCES}" "${BENCHMARK_HEADERS}")

set(LIBS
	${Boost_LIBRARIES}
	${benchmark_LIBRARY}
	Shlwapi
)

target_link_libraries(VectorizeAVX ${LIBS})

target_compile_options(VectorizeAVX PRIVATE /arch:AVX)

target_compile_definitions(VectorizeAVX PRIVATE 
	EIGEN_NO_DEBUG=1)

set_target_properties(VectorizeAVX PROPERTIES FOLDER "Benchmarks")

