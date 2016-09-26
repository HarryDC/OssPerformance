find_package(Boost 1.54 COMPONENTS program_options REQUIRED)

include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}
	"${Boost_INCLUDE_DIR}"
	"${EIGEN3_INCLUDE_DIR}"
	"${OPENSURGSIM_INCLUDE_DIRS}"
	"${OSG_INCLUDE_DIR}"
	"${YAML_CPP_INCLUDE_DIR}"
)

set(BENCHMARK_SOURCES
	Collision.cpp		
)

set(BENCHMARK_HEADERS

)

surgsim_add_executable(Collision "${BENCHMARK_SOURCES}" "${BENCHMARK_HEADERS}")

set(LIBS
	${Boost_LIBRARIES}
	${benchmark_LIBRARY}
	Shlwapi
	SurgSimCollision
)

target_link_libraries(Collision ${LIBS})

set_target_properties(Collision PROPERTIES FOLDER "Benchmarks")