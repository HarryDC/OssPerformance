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
)

set(BENCHMARK_HEADERS

)

surgsim_add_executable(VectorizeSSE "${BENCHMARK_SOURCES}" "${BENCHMARK_HEADERS}")

set(LIBS
	${Boost_LIBRARIES}
	${benchmark_LIBRARY}
	Shlwapi
)



target_compile_definitions(VectorizeSSE PRIVATE 
	EIGEN_NO_DEBUG=1
	EIGEN_VECTORIZE_SSE4_2=1)

target_link_libraries(VectorizeSSE ${LIBS})

set_target_properties(VectorizeSSE PROPERTIES FOLDER "Benchmarks")

