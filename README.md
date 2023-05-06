# Summary

We created three 3D Conway's Game of Life implementations: sequential C++, parallel CUDA, and parallel OpenMP to compare the speedup. We also created a renderer using raylib, an OpenGL wrapper, to visualize our 3D results.

# Report

A detailed final report with our approach and results can be found on our website [here](https://kat-04.github.io).

# Future Work

- Speed up rendering time by culling voxels not visible
- Add ambient occlusion based on number of neighbors to easily distinguish depth of voxels
- Oct-tree implementation for OpenMP
- CUDA bounding box speedup
- Parsing input in batches to overcome CUDA memory limit

# Installation/Run Process

## Main Algorithm

1. Clone the repository
2. Create a `CMakeLists.txt` file that compiles in the main folder and another one in the `render/` folder. Note that you may have to `brew install llvm` for OpenMP. See below for details.
3. To run the executable `./GOL3D` that is generated, the following inputs are accepted:
```
./GOL3D [input_file_name] [num_frames] [side_length] [seq(default)/cuda/omp/states]
```

## Generating Test Cases

Some interesting test cases can be found under the `input-files/` directory. There is also a python file, `fill_box.py` to generate new randomized test cases. The run command is as follows:
```
python3 fill_box.py [input_file] -r [rule] -o [offset (default 0)] -n [side_length] -s [starting state (default none)] -d [density (default 0.5)]
```

## Rendering

The render that we used uses the OpenGL wrapper `raylib`. In the `render` folder, create a new directory called `libs`. Inside `libs`, clone the repository linked [here](https://github.com/raysan5/raylib).

As our render uses lights and shaders, you must first move `rlights.h` from `raylib/examples/shaders/rlights.h` to `raylib/src/`. Then, in the `CMakeLists.txt` file inside the `raylib/src/` directory, add `rlights.h` under line 24: `set (raylib_public_headers ...)`. You do not need to run `cmake` in the folder.

Creatw the `CMakeLists.txt` file in the `render` folder (see below for details), and run the command `cmake .`. This should generate the `Makefile` for your computer.

You should then be able to run `make` within the folder then run the executable `./GOL3D` that is created after.

If issues occur, then you may need to run one of the following commands:

1) `brew install glfw3`
2) `brew install glew`
3) `brew install pkg-config`

## How to navigate the window

So after you run `./GOL3D`, a window will pop up that has a rotating voxel cube in the middle. The red wireframe border is the "bounding box" and the cubes inside are the cells.

The initial state of the scene is with an orbital camera and automatic frame increments (set to every 1 seconds).

### Toggle Options

To change the camera mode between free/orbiting, press `C` for camera.

To change the automatic frames to manual frames (mainly for debugging), press `F` for frame. To reset the frames to frame 0 (beginning), press `R` for reset.

To view the bounding box, press `B` for box (default is off).

### Navigation

For the free camera (3rd person), there are many different keys that manage the camera rotation and panning:
- `WASD` pans the camera (`A` = camera to the left, `D` = to the right, `W` = forward, `S` = backward)
- `Q` rotates the cube clockwise and `E` rotates the cube counterclockwise.
- `Up, Down, Left, Right` arrow keys rotate the camera in that direction
- Using 2 fingers to slide up on the trackpad zooms in, slide down zooms out
- Moving the cursor within the window rotates the object with respect to your cursor location

For the frames:
- To go one frame ahead, press `.`
- To move back a frame, press `,`

## Other resources

You can find the GitHub for raylib [here](https://github.com/raysan5/raylib) but examples may be more helpful and can be found [here](https://www.raylib.com/examples.html).

# CMakeLists.txt for MacOS

In the main folder:
```
cmake_minimum_required(VERSION 3.9)
project(GOL3D LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_COMPILER "/usr/local/Cellar/llvm/16.0.2/bin/clang++")
set(OPENMP_LIBRARIES "/usr/local/Cellar/llvm/16.0.2/lib")
set(OPENMP_INCLUDES "/usr/local/Cellar/llvm/16.0.2/include")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra -fopenmp")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_executable(${PROJECT_NAME} src/main-local.cpp src/gol-sequential.cpp src/gol-openmp.cpp src/gol-openmp-states.cpp src/parse.cpp)
```

If you have CUDA, in the main you can add:
```
project(GOL3D LANGUAGES CXX CUDA) # instead of project(GOL3D LANGUAGES CXX)
set_property(TARGET GOL3D PROPERTY CUDA_ARCHITECTURES 61) # change value to match your architecture

add_executable(${PROJECT_NAME} src/main-local.cpp src/gol-sequential.cpp src/golCuda.cu src/gol-openmp.cpp src/gol-openmp-states.cpp src/parse.cpp)
```

In render:
```
cmake_minimum_required(VERSION 3.0)
project(GOL3D C CXX)

set(CMAKE_CXX_STANDARD 17)

# Setting parameters for raylib
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE) # don't build the supplied examples
set(BUILD_GAMES    OFF CACHE BOOL "" FORCE) # or games

add_subdirectory(libs/raylib)

add_executable(${PROJECT_NAME} src/main-local.cpp src/mesh.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE raylib)
target_compile_definitions(${PROJECT_NAME} PUBLIC ASSETS_PATH="${CMAKE_CURRENT_SOURCE_DIR}/assets/")
```
