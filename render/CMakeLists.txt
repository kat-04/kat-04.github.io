cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
project(GOL3D)

# Add source files
# file (GLOB_RECURSE SOURCE_FILES
#       ${CMAKE_SOURCE_DIR}/*.cpp)

# Do the same for header files if used

include_directories(/usr/local/include)
link_directories(/usr/local/lib)

add_executable(GOL3D render_cube.cpp)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
include_directories( ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS} )

add_compile_options("-g")

target_link_libraries(GOL3D ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} )
target_link_libraries(GOL3D glfw)
target_link_libraries(GOL3D glew)

