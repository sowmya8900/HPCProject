# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/sowmya/Documents/Learn/HPCProject/PDE_Solvers

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/sowmya/Documents/Learn/HPCProject/PDE_Solvers/build

# Include any dependencies generated for this target.
include CMakeFiles/heat_solver.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/heat_solver.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/heat_solver.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/heat_solver.dir/flags.make

CMakeFiles/heat_solver.dir/src/heat_solver.cpp.o: CMakeFiles/heat_solver.dir/flags.make
CMakeFiles/heat_solver.dir/src/heat_solver.cpp.o: /home/sowmya/Documents/Learn/HPCProject/PDE_Solvers/src/heat_solver.cpp
CMakeFiles/heat_solver.dir/src/heat_solver.cpp.o: CMakeFiles/heat_solver.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/sowmya/Documents/Learn/HPCProject/PDE_Solvers/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/heat_solver.dir/src/heat_solver.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/heat_solver.dir/src/heat_solver.cpp.o -MF CMakeFiles/heat_solver.dir/src/heat_solver.cpp.o.d -o CMakeFiles/heat_solver.dir/src/heat_solver.cpp.o -c /home/sowmya/Documents/Learn/HPCProject/PDE_Solvers/src/heat_solver.cpp

CMakeFiles/heat_solver.dir/src/heat_solver.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/heat_solver.dir/src/heat_solver.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/sowmya/Documents/Learn/HPCProject/PDE_Solvers/src/heat_solver.cpp > CMakeFiles/heat_solver.dir/src/heat_solver.cpp.i

CMakeFiles/heat_solver.dir/src/heat_solver.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/heat_solver.dir/src/heat_solver.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/sowmya/Documents/Learn/HPCProject/PDE_Solvers/src/heat_solver.cpp -o CMakeFiles/heat_solver.dir/src/heat_solver.cpp.s

# Object files for target heat_solver
heat_solver_OBJECTS = \
"CMakeFiles/heat_solver.dir/src/heat_solver.cpp.o"

# External object files for target heat_solver
heat_solver_EXTERNAL_OBJECTS =

heat_solver: CMakeFiles/heat_solver.dir/src/heat_solver.cpp.o
heat_solver: CMakeFiles/heat_solver.dir/build.make
heat_solver: /usr/lib/gcc/x86_64-linux-gnu/13/libgomp.so
heat_solver: /usr/lib/x86_64-linux-gnu/libpthread.a
heat_solver: CMakeFiles/heat_solver.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/sowmya/Documents/Learn/HPCProject/PDE_Solvers/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable heat_solver"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/heat_solver.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/heat_solver.dir/build: heat_solver
.PHONY : CMakeFiles/heat_solver.dir/build

CMakeFiles/heat_solver.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/heat_solver.dir/cmake_clean.cmake
.PHONY : CMakeFiles/heat_solver.dir/clean

CMakeFiles/heat_solver.dir/depend:
	cd /home/sowmya/Documents/Learn/HPCProject/PDE_Solvers/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/sowmya/Documents/Learn/HPCProject/PDE_Solvers /home/sowmya/Documents/Learn/HPCProject/PDE_Solvers /home/sowmya/Documents/Learn/HPCProject/PDE_Solvers/build /home/sowmya/Documents/Learn/HPCProject/PDE_Solvers/build /home/sowmya/Documents/Learn/HPCProject/PDE_Solvers/build/CMakeFiles/heat_solver.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/heat_solver.dir/depend

