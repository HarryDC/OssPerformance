
find_package(OpenCL REQUIRED)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

link_directories(
	${Boost_LIBRARY_DIRS}
)


include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}
	"${Boost_INCLUDE_DIR}"
	"${EIGEN3_INCLUDE_DIR}"
	"${OPENSURGSIM_INCLUDE_DIRS}"
	"${OSG_INCLUDE_DIR}"
	"${YAML_CPP_INCLUDE_DIR}"
	"${OpenCL_INCLUDE_DIR}"
	${VIENNACL_INCLUDE_DIRS}
)

set(BENCHMARK_SOURCES
	Compute.cpp
	ComputeMain.cpp
)

set(BENCHMARK_HEADERS

)

surgsim_add_executable(Compute "${BENCHMARK_SOURCES}" "${BENCHMARK_HEADERS}")

set(LIBS
	${Boost_LIBRARIES}
	${benchmark_LIBRARY}
	Shlwapi
	${OpenCL_LIBRARIES}
	SurgSimFramework
)

target_link_libraries(Compute ${LIBS})

set_target_properties(Compute PROPERTIES FOLDER "Benchmarks")
