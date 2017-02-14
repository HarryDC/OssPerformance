find_package(Boost 1.54 COMPONENTS program_options REQUIRED)

include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}
	"${Boost_INCLUDE_DIR}"
	"${EIGEN3_INCLUDE_DIR}"
	"${OPENSURGSIM_INCLUDE_DIRS}"
	"${OSG_INCLUDE_DIR}"
	"${YAML_CPP_INCLUDE_DIR}"
)

set(DEFORMABLES_SOURCES
    Deformables.cpp
	Scene.cpp
)

set(DEFORMABLES_HEADERS
	Scene.h
)

# Configure the path for the data files
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/config.txt.in"
    "${CMAKE_CURRENT_BINARY_DIR}/config.txt"
)

surgsim_add_executable(Deformables "${DEFORMABLES_SOURCES}" "${DEFORMABLES_HEADERS}")

set(LIBS
	${Boost_LIBRARIES}
	SurgSimBlocks
	SurgSimCollision
	SurgSimDataStructures
	SurgSimFramework
	SurgSimGraphics
	SurgSimMath
	SurgSimPhysics
	SurgSimParallel
)

target_link_libraries(Deformables ${LIBS})

set_target_properties(Deformables PROPERTIES FOLDER "Performance")