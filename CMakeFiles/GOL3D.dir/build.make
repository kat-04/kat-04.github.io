# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.26

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Applications/CMake.app/Contents/bin/cmake

# The command to remove a file.
RM = /Applications/CMake.app/Contents/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/katherinelu/Documents/15418/kat-04.github.io

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/katherinelu/Documents/15418/kat-04.github.io

# Include any dependencies generated for this target.
include CMakeFiles/GOL3D.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/GOL3D.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/GOL3D.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/GOL3D.dir/flags.make

CMakeFiles/GOL3D.dir/src/main-local.cpp.o: CMakeFiles/GOL3D.dir/flags.make
CMakeFiles/GOL3D.dir/src/main-local.cpp.o: src/main-local.cpp
CMakeFiles/GOL3D.dir/src/main-local.cpp.o: CMakeFiles/GOL3D.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/katherinelu/Documents/15418/kat-04.github.io/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/GOL3D.dir/src/main-local.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/GOL3D.dir/src/main-local.cpp.o -MF CMakeFiles/GOL3D.dir/src/main-local.cpp.o.d -o CMakeFiles/GOL3D.dir/src/main-local.cpp.o -c /Users/katherinelu/Documents/15418/kat-04.github.io/src/main-local.cpp

CMakeFiles/GOL3D.dir/src/main-local.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/GOL3D.dir/src/main-local.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/katherinelu/Documents/15418/kat-04.github.io/src/main-local.cpp > CMakeFiles/GOL3D.dir/src/main-local.cpp.i

CMakeFiles/GOL3D.dir/src/main-local.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/GOL3D.dir/src/main-local.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/katherinelu/Documents/15418/kat-04.github.io/src/main-local.cpp -o CMakeFiles/GOL3D.dir/src/main-local.cpp.s

CMakeFiles/GOL3D.dir/src/gol-sequential.cpp.o: CMakeFiles/GOL3D.dir/flags.make
CMakeFiles/GOL3D.dir/src/gol-sequential.cpp.o: src/gol-sequential.cpp
CMakeFiles/GOL3D.dir/src/gol-sequential.cpp.o: CMakeFiles/GOL3D.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/katherinelu/Documents/15418/kat-04.github.io/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/GOL3D.dir/src/gol-sequential.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/GOL3D.dir/src/gol-sequential.cpp.o -MF CMakeFiles/GOL3D.dir/src/gol-sequential.cpp.o.d -o CMakeFiles/GOL3D.dir/src/gol-sequential.cpp.o -c /Users/katherinelu/Documents/15418/kat-04.github.io/src/gol-sequential.cpp

CMakeFiles/GOL3D.dir/src/gol-sequential.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/GOL3D.dir/src/gol-sequential.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/katherinelu/Documents/15418/kat-04.github.io/src/gol-sequential.cpp > CMakeFiles/GOL3D.dir/src/gol-sequential.cpp.i

CMakeFiles/GOL3D.dir/src/gol-sequential.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/GOL3D.dir/src/gol-sequential.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/katherinelu/Documents/15418/kat-04.github.io/src/gol-sequential.cpp -o CMakeFiles/GOL3D.dir/src/gol-sequential.cpp.s

# Object files for target GOL3D
GOL3D_OBJECTS = \
"CMakeFiles/GOL3D.dir/src/main-local.cpp.o" \
"CMakeFiles/GOL3D.dir/src/gol-sequential.cpp.o"

# External object files for target GOL3D
GOL3D_EXTERNAL_OBJECTS =

GOL3D: CMakeFiles/GOL3D.dir/src/main-local.cpp.o
GOL3D: CMakeFiles/GOL3D.dir/src/gol-sequential.cpp.o
GOL3D: CMakeFiles/GOL3D.dir/build.make
GOL3D: CMakeFiles/GOL3D.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/katherinelu/Documents/15418/kat-04.github.io/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable GOL3D"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/GOL3D.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/GOL3D.dir/build: GOL3D
.PHONY : CMakeFiles/GOL3D.dir/build

CMakeFiles/GOL3D.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/GOL3D.dir/cmake_clean.cmake
.PHONY : CMakeFiles/GOL3D.dir/clean

CMakeFiles/GOL3D.dir/depend:
	cd /Users/katherinelu/Documents/15418/kat-04.github.io && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/katherinelu/Documents/15418/kat-04.github.io /Users/katherinelu/Documents/15418/kat-04.github.io /Users/katherinelu/Documents/15418/kat-04.github.io /Users/katherinelu/Documents/15418/kat-04.github.io /Users/katherinelu/Documents/15418/kat-04.github.io/CMakeFiles/GOL3D.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/GOL3D.dir/depend

