
find_package(OpenCL REQUIRED)
find_package(ViennaCL REQUIRED)

add_definitions( -DVIENNACL_WITH_EIGEN )
add_definitions( -DVIENNACL_WITH_OPENCL )

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(TBB REQUIRED)

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
	${TBB_INCLUDE_DIRS}
)

set(BENCHMARK_SOURCES
	Compute.cpp
	ComputeMain.cpp
	TBB.cpp
	ViennaCl.cpp
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
	${TBB_LIBRARIES}
)

target_link_libraries(Compute ${LIBS})

set_target_properties(Compute PROPERTIES FOLDER "Benchmarks")
target_compile_definitions(Compute PRIVATE ${TBB_DEFINITIONS})
