# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.7

# Default target executed when no arguments are given to make.
default_target: all

.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
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
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/the-user/Downloads/cadproj-01

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/the-user/Downloads/cadproj-01

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/bin/cmake -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache

.PHONY : rebuild_cache/fast

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "No interactive CMake dialog available..."
	/usr/bin/cmake -E echo No\ interactive\ CMake\ dialog\ available.
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache

.PHONY : edit_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/the-user/Downloads/cadproj-01/CMakeFiles /home/the-user/Downloads/cadproj-01/CMakeFiles/progress.marks
	$(MAKE) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/the-user/Downloads/cadproj-01/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean

.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named wb

# Build rule for target.
wb: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 wb
.PHONY : wb

# fast build rule for target.
wb/fast:
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/build
.PHONY : wb/fast

#=============================================================================
# Target rules for targets named histogram_equalization

# Build rule for target.
histogram_equalization: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 histogram_equalization
.PHONY : histogram_equalization

# fast build rule for target.
histogram_equalization/fast:
	$(MAKE) -f CMakeFiles/histogram_equalization.dir/build.make CMakeFiles/histogram_equalization.dir/build
.PHONY : histogram_equalization/fast

#=============================================================================
# Target rules for targets named clean_cuda_depends

# Build rule for target.
clean_cuda_depends: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 clean_cuda_depends
.PHONY : clean_cuda_depends

# fast build rule for target.
clean_cuda_depends/fast:
	$(MAKE) -f CMakeFiles/clean_cuda_depends.dir/build.make CMakeFiles/clean_cuda_depends.dir/build
.PHONY : clean_cuda_depends/fast

home/the-user/Desktop/CAD/TP1/libwb/vendor/json11.o: home/the-user/Desktop/CAD/TP1/libwb/vendor/json11.cpp.o

.PHONY : home/the-user/Desktop/CAD/TP1/libwb/vendor/json11.o

# target to build an object file
home/the-user/Desktop/CAD/TP1/libwb/vendor/json11.cpp.o:
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/home/the-user/Desktop/CAD/TP1/libwb/vendor/json11.cpp.o
.PHONY : home/the-user/Desktop/CAD/TP1/libwb/vendor/json11.cpp.o

home/the-user/Desktop/CAD/TP1/libwb/vendor/json11.i: home/the-user/Desktop/CAD/TP1/libwb/vendor/json11.cpp.i

.PHONY : home/the-user/Desktop/CAD/TP1/libwb/vendor/json11.i

# target to preprocess a source file
home/the-user/Desktop/CAD/TP1/libwb/vendor/json11.cpp.i:
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/home/the-user/Desktop/CAD/TP1/libwb/vendor/json11.cpp.i
.PHONY : home/the-user/Desktop/CAD/TP1/libwb/vendor/json11.cpp.i

home/the-user/Desktop/CAD/TP1/libwb/vendor/json11.s: home/the-user/Desktop/CAD/TP1/libwb/vendor/json11.cpp.s

.PHONY : home/the-user/Desktop/CAD/TP1/libwb/vendor/json11.s

# target to generate assembly for a file
home/the-user/Desktop/CAD/TP1/libwb/vendor/json11.cpp.s:
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/home/the-user/Desktop/CAD/TP1/libwb/vendor/json11.cpp.s
.PHONY : home/the-user/Desktop/CAD/TP1/libwb/vendor/json11.cpp.s

home/the-user/Desktop/CAD/TP1/libwb/wbArg.o: home/the-user/Desktop/CAD/TP1/libwb/wbArg.cpp.o

.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbArg.o

# target to build an object file
home/the-user/Desktop/CAD/TP1/libwb/wbArg.cpp.o:
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/home/the-user/Desktop/CAD/TP1/libwb/wbArg.cpp.o
.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbArg.cpp.o

home/the-user/Desktop/CAD/TP1/libwb/wbArg.i: home/the-user/Desktop/CAD/TP1/libwb/wbArg.cpp.i

.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbArg.i

# target to preprocess a source file
home/the-user/Desktop/CAD/TP1/libwb/wbArg.cpp.i:
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/home/the-user/Desktop/CAD/TP1/libwb/wbArg.cpp.i
.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbArg.cpp.i

home/the-user/Desktop/CAD/TP1/libwb/wbArg.s: home/the-user/Desktop/CAD/TP1/libwb/wbArg.cpp.s

.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbArg.s

# target to generate assembly for a file
home/the-user/Desktop/CAD/TP1/libwb/wbArg.cpp.s:
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/home/the-user/Desktop/CAD/TP1/libwb/wbArg.cpp.s
.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbArg.cpp.s

home/the-user/Desktop/CAD/TP1/libwb/wbCUDA.o: home/the-user/Desktop/CAD/TP1/libwb/wbCUDA.cpp.o

.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbCUDA.o

# target to build an object file
home/the-user/Desktop/CAD/TP1/libwb/wbCUDA.cpp.o:
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/home/the-user/Desktop/CAD/TP1/libwb/wbCUDA.cpp.o
.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbCUDA.cpp.o

home/the-user/Desktop/CAD/TP1/libwb/wbCUDA.i: home/the-user/Desktop/CAD/TP1/libwb/wbCUDA.cpp.i

.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbCUDA.i

# target to preprocess a source file
home/the-user/Desktop/CAD/TP1/libwb/wbCUDA.cpp.i:
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/home/the-user/Desktop/CAD/TP1/libwb/wbCUDA.cpp.i
.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbCUDA.cpp.i

home/the-user/Desktop/CAD/TP1/libwb/wbCUDA.s: home/the-user/Desktop/CAD/TP1/libwb/wbCUDA.cpp.s

.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbCUDA.s

# target to generate assembly for a file
home/the-user/Desktop/CAD/TP1/libwb/wbCUDA.cpp.s:
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/home/the-user/Desktop/CAD/TP1/libwb/wbCUDA.cpp.s
.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbCUDA.cpp.s

home/the-user/Desktop/CAD/TP1/libwb/wbDataset.o: home/the-user/Desktop/CAD/TP1/libwb/wbDataset.cpp.o

.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbDataset.o

# target to build an object file
home/the-user/Desktop/CAD/TP1/libwb/wbDataset.cpp.o:
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/home/the-user/Desktop/CAD/TP1/libwb/wbDataset.cpp.o
.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbDataset.cpp.o

home/the-user/Desktop/CAD/TP1/libwb/wbDataset.i: home/the-user/Desktop/CAD/TP1/libwb/wbDataset.cpp.i

.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbDataset.i

# target to preprocess a source file
home/the-user/Desktop/CAD/TP1/libwb/wbDataset.cpp.i:
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/home/the-user/Desktop/CAD/TP1/libwb/wbDataset.cpp.i
.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbDataset.cpp.i

home/the-user/Desktop/CAD/TP1/libwb/wbDataset.s: home/the-user/Desktop/CAD/TP1/libwb/wbDataset.cpp.s

.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbDataset.s

# target to generate assembly for a file
home/the-user/Desktop/CAD/TP1/libwb/wbDataset.cpp.s:
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/home/the-user/Desktop/CAD/TP1/libwb/wbDataset.cpp.s
.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbDataset.cpp.s

home/the-user/Desktop/CAD/TP1/libwb/wbDirectory.o: home/the-user/Desktop/CAD/TP1/libwb/wbDirectory.cpp.o

.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbDirectory.o

# target to build an object file
home/the-user/Desktop/CAD/TP1/libwb/wbDirectory.cpp.o:
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/home/the-user/Desktop/CAD/TP1/libwb/wbDirectory.cpp.o
.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbDirectory.cpp.o

home/the-user/Desktop/CAD/TP1/libwb/wbDirectory.i: home/the-user/Desktop/CAD/TP1/libwb/wbDirectory.cpp.i

.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbDirectory.i

# target to preprocess a source file
home/the-user/Desktop/CAD/TP1/libwb/wbDirectory.cpp.i:
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/home/the-user/Desktop/CAD/TP1/libwb/wbDirectory.cpp.i
.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbDirectory.cpp.i

home/the-user/Desktop/CAD/TP1/libwb/wbDirectory.s: home/the-user/Desktop/CAD/TP1/libwb/wbDirectory.cpp.s

.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbDirectory.s

# target to generate assembly for a file
home/the-user/Desktop/CAD/TP1/libwb/wbDirectory.cpp.s:
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/home/the-user/Desktop/CAD/TP1/libwb/wbDirectory.cpp.s
.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbDirectory.cpp.s

home/the-user/Desktop/CAD/TP1/libwb/wbExit.o: home/the-user/Desktop/CAD/TP1/libwb/wbExit.cpp.o

.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbExit.o

# target to build an object file
home/the-user/Desktop/CAD/TP1/libwb/wbExit.cpp.o:
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/home/the-user/Desktop/CAD/TP1/libwb/wbExit.cpp.o
.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbExit.cpp.o

home/the-user/Desktop/CAD/TP1/libwb/wbExit.i: home/the-user/Desktop/CAD/TP1/libwb/wbExit.cpp.i

.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbExit.i

# target to preprocess a source file
home/the-user/Desktop/CAD/TP1/libwb/wbExit.cpp.i:
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/home/the-user/Desktop/CAD/TP1/libwb/wbExit.cpp.i
.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbExit.cpp.i

home/the-user/Desktop/CAD/TP1/libwb/wbExit.s: home/the-user/Desktop/CAD/TP1/libwb/wbExit.cpp.s

.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbExit.s

# target to generate assembly for a file
home/the-user/Desktop/CAD/TP1/libwb/wbExit.cpp.s:
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/home/the-user/Desktop/CAD/TP1/libwb/wbExit.cpp.s
.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbExit.cpp.s

home/the-user/Desktop/CAD/TP1/libwb/wbExport.o: home/the-user/Desktop/CAD/TP1/libwb/wbExport.cpp.o

.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbExport.o

# target to build an object file
home/the-user/Desktop/CAD/TP1/libwb/wbExport.cpp.o:
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/home/the-user/Desktop/CAD/TP1/libwb/wbExport.cpp.o
.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbExport.cpp.o

home/the-user/Desktop/CAD/TP1/libwb/wbExport.i: home/the-user/Desktop/CAD/TP1/libwb/wbExport.cpp.i

.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbExport.i

# target to preprocess a source file
home/the-user/Desktop/CAD/TP1/libwb/wbExport.cpp.i:
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/home/the-user/Desktop/CAD/TP1/libwb/wbExport.cpp.i
.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbExport.cpp.i

home/the-user/Desktop/CAD/TP1/libwb/wbExport.s: home/the-user/Desktop/CAD/TP1/libwb/wbExport.cpp.s

.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbExport.s

# target to generate assembly for a file
home/the-user/Desktop/CAD/TP1/libwb/wbExport.cpp.s:
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/home/the-user/Desktop/CAD/TP1/libwb/wbExport.cpp.s
.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbExport.cpp.s

home/the-user/Desktop/CAD/TP1/libwb/wbFile.o: home/the-user/Desktop/CAD/TP1/libwb/wbFile.cpp.o

.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbFile.o

# target to build an object file
home/the-user/Desktop/CAD/TP1/libwb/wbFile.cpp.o:
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/home/the-user/Desktop/CAD/TP1/libwb/wbFile.cpp.o
.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbFile.cpp.o

home/the-user/Desktop/CAD/TP1/libwb/wbFile.i: home/the-user/Desktop/CAD/TP1/libwb/wbFile.cpp.i

.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbFile.i

# target to preprocess a source file
home/the-user/Desktop/CAD/TP1/libwb/wbFile.cpp.i:
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/home/the-user/Desktop/CAD/TP1/libwb/wbFile.cpp.i
.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbFile.cpp.i

home/the-user/Desktop/CAD/TP1/libwb/wbFile.s: home/the-user/Desktop/CAD/TP1/libwb/wbFile.cpp.s

.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbFile.s

# target to generate assembly for a file
home/the-user/Desktop/CAD/TP1/libwb/wbFile.cpp.s:
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/home/the-user/Desktop/CAD/TP1/libwb/wbFile.cpp.s
.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbFile.cpp.s

home/the-user/Desktop/CAD/TP1/libwb/wbImage.o: home/the-user/Desktop/CAD/TP1/libwb/wbImage.cpp.o

.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbImage.o

# target to build an object file
home/the-user/Desktop/CAD/TP1/libwb/wbImage.cpp.o:
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/home/the-user/Desktop/CAD/TP1/libwb/wbImage.cpp.o
.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbImage.cpp.o

home/the-user/Desktop/CAD/TP1/libwb/wbImage.i: home/the-user/Desktop/CAD/TP1/libwb/wbImage.cpp.i

.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbImage.i

# target to preprocess a source file
home/the-user/Desktop/CAD/TP1/libwb/wbImage.cpp.i:
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/home/the-user/Desktop/CAD/TP1/libwb/wbImage.cpp.i
.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbImage.cpp.i

home/the-user/Desktop/CAD/TP1/libwb/wbImage.s: home/the-user/Desktop/CAD/TP1/libwb/wbImage.cpp.s

.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbImage.s

# target to generate assembly for a file
home/the-user/Desktop/CAD/TP1/libwb/wbImage.cpp.s:
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/home/the-user/Desktop/CAD/TP1/libwb/wbImage.cpp.s
.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbImage.cpp.s

home/the-user/Desktop/CAD/TP1/libwb/wbImport.o: home/the-user/Desktop/CAD/TP1/libwb/wbImport.cpp.o

.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbImport.o

# target to build an object file
home/the-user/Desktop/CAD/TP1/libwb/wbImport.cpp.o:
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/home/the-user/Desktop/CAD/TP1/libwb/wbImport.cpp.o
.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbImport.cpp.o

home/the-user/Desktop/CAD/TP1/libwb/wbImport.i: home/the-user/Desktop/CAD/TP1/libwb/wbImport.cpp.i

.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbImport.i

# target to preprocess a source file
home/the-user/Desktop/CAD/TP1/libwb/wbImport.cpp.i:
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/home/the-user/Desktop/CAD/TP1/libwb/wbImport.cpp.i
.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbImport.cpp.i

home/the-user/Desktop/CAD/TP1/libwb/wbImport.s: home/the-user/Desktop/CAD/TP1/libwb/wbImport.cpp.s

.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbImport.s

# target to generate assembly for a file
home/the-user/Desktop/CAD/TP1/libwb/wbImport.cpp.s:
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/home/the-user/Desktop/CAD/TP1/libwb/wbImport.cpp.s
.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbImport.cpp.s

home/the-user/Desktop/CAD/TP1/libwb/wbInit.o: home/the-user/Desktop/CAD/TP1/libwb/wbInit.cpp.o

.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbInit.o

# target to build an object file
home/the-user/Desktop/CAD/TP1/libwb/wbInit.cpp.o:
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/home/the-user/Desktop/CAD/TP1/libwb/wbInit.cpp.o
.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbInit.cpp.o

home/the-user/Desktop/CAD/TP1/libwb/wbInit.i: home/the-user/Desktop/CAD/TP1/libwb/wbInit.cpp.i

.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbInit.i

# target to preprocess a source file
home/the-user/Desktop/CAD/TP1/libwb/wbInit.cpp.i:
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/home/the-user/Desktop/CAD/TP1/libwb/wbInit.cpp.i
.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbInit.cpp.i

home/the-user/Desktop/CAD/TP1/libwb/wbInit.s: home/the-user/Desktop/CAD/TP1/libwb/wbInit.cpp.s

.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbInit.s

# target to generate assembly for a file
home/the-user/Desktop/CAD/TP1/libwb/wbInit.cpp.s:
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/home/the-user/Desktop/CAD/TP1/libwb/wbInit.cpp.s
.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbInit.cpp.s

home/the-user/Desktop/CAD/TP1/libwb/wbLogger.o: home/the-user/Desktop/CAD/TP1/libwb/wbLogger.cpp.o

.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbLogger.o

# target to build an object file
home/the-user/Desktop/CAD/TP1/libwb/wbLogger.cpp.o:
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/home/the-user/Desktop/CAD/TP1/libwb/wbLogger.cpp.o
.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbLogger.cpp.o

home/the-user/Desktop/CAD/TP1/libwb/wbLogger.i: home/the-user/Desktop/CAD/TP1/libwb/wbLogger.cpp.i

.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbLogger.i

# target to preprocess a source file
home/the-user/Desktop/CAD/TP1/libwb/wbLogger.cpp.i:
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/home/the-user/Desktop/CAD/TP1/libwb/wbLogger.cpp.i
.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbLogger.cpp.i

home/the-user/Desktop/CAD/TP1/libwb/wbLogger.s: home/the-user/Desktop/CAD/TP1/libwb/wbLogger.cpp.s

.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbLogger.s

# target to generate assembly for a file
home/the-user/Desktop/CAD/TP1/libwb/wbLogger.cpp.s:
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/home/the-user/Desktop/CAD/TP1/libwb/wbLogger.cpp.s
.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbLogger.cpp.s

home/the-user/Desktop/CAD/TP1/libwb/wbMPI.o: home/the-user/Desktop/CAD/TP1/libwb/wbMPI.cpp.o

.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbMPI.o

# target to build an object file
home/the-user/Desktop/CAD/TP1/libwb/wbMPI.cpp.o:
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/home/the-user/Desktop/CAD/TP1/libwb/wbMPI.cpp.o
.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbMPI.cpp.o

home/the-user/Desktop/CAD/TP1/libwb/wbMPI.i: home/the-user/Desktop/CAD/TP1/libwb/wbMPI.cpp.i

.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbMPI.i

# target to preprocess a source file
home/the-user/Desktop/CAD/TP1/libwb/wbMPI.cpp.i:
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/home/the-user/Desktop/CAD/TP1/libwb/wbMPI.cpp.i
.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbMPI.cpp.i

home/the-user/Desktop/CAD/TP1/libwb/wbMPI.s: home/the-user/Desktop/CAD/TP1/libwb/wbMPI.cpp.s

.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbMPI.s

# target to generate assembly for a file
home/the-user/Desktop/CAD/TP1/libwb/wbMPI.cpp.s:
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/home/the-user/Desktop/CAD/TP1/libwb/wbMPI.cpp.s
.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbMPI.cpp.s

home/the-user/Desktop/CAD/TP1/libwb/wbPPM.o: home/the-user/Desktop/CAD/TP1/libwb/wbPPM.cpp.o

.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbPPM.o

# target to build an object file
home/the-user/Desktop/CAD/TP1/libwb/wbPPM.cpp.o:
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/home/the-user/Desktop/CAD/TP1/libwb/wbPPM.cpp.o
.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbPPM.cpp.o

home/the-user/Desktop/CAD/TP1/libwb/wbPPM.i: home/the-user/Desktop/CAD/TP1/libwb/wbPPM.cpp.i

.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbPPM.i

# target to preprocess a source file
home/the-user/Desktop/CAD/TP1/libwb/wbPPM.cpp.i:
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/home/the-user/Desktop/CAD/TP1/libwb/wbPPM.cpp.i
.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbPPM.cpp.i

home/the-user/Desktop/CAD/TP1/libwb/wbPPM.s: home/the-user/Desktop/CAD/TP1/libwb/wbPPM.cpp.s

.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbPPM.s

# target to generate assembly for a file
home/the-user/Desktop/CAD/TP1/libwb/wbPPM.cpp.s:
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/home/the-user/Desktop/CAD/TP1/libwb/wbPPM.cpp.s
.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbPPM.cpp.s

home/the-user/Desktop/CAD/TP1/libwb/wbPath.o: home/the-user/Desktop/CAD/TP1/libwb/wbPath.cpp.o

.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbPath.o

# target to build an object file
home/the-user/Desktop/CAD/TP1/libwb/wbPath.cpp.o:
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/home/the-user/Desktop/CAD/TP1/libwb/wbPath.cpp.o
.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbPath.cpp.o

home/the-user/Desktop/CAD/TP1/libwb/wbPath.i: home/the-user/Desktop/CAD/TP1/libwb/wbPath.cpp.i

.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbPath.i

# target to preprocess a source file
home/the-user/Desktop/CAD/TP1/libwb/wbPath.cpp.i:
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/home/the-user/Desktop/CAD/TP1/libwb/wbPath.cpp.i
.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbPath.cpp.i

home/the-user/Desktop/CAD/TP1/libwb/wbPath.s: home/the-user/Desktop/CAD/TP1/libwb/wbPath.cpp.s

.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbPath.s

# target to generate assembly for a file
home/the-user/Desktop/CAD/TP1/libwb/wbPath.cpp.s:
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/home/the-user/Desktop/CAD/TP1/libwb/wbPath.cpp.s
.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbPath.cpp.s

home/the-user/Desktop/CAD/TP1/libwb/wbSolution.o: home/the-user/Desktop/CAD/TP1/libwb/wbSolution.cpp.o

.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbSolution.o

# target to build an object file
home/the-user/Desktop/CAD/TP1/libwb/wbSolution.cpp.o:
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/home/the-user/Desktop/CAD/TP1/libwb/wbSolution.cpp.o
.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbSolution.cpp.o

home/the-user/Desktop/CAD/TP1/libwb/wbSolution.i: home/the-user/Desktop/CAD/TP1/libwb/wbSolution.cpp.i

.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbSolution.i

# target to preprocess a source file
home/the-user/Desktop/CAD/TP1/libwb/wbSolution.cpp.i:
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/home/the-user/Desktop/CAD/TP1/libwb/wbSolution.cpp.i
.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbSolution.cpp.i

home/the-user/Desktop/CAD/TP1/libwb/wbSolution.s: home/the-user/Desktop/CAD/TP1/libwb/wbSolution.cpp.s

.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbSolution.s

# target to generate assembly for a file
home/the-user/Desktop/CAD/TP1/libwb/wbSolution.cpp.s:
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/home/the-user/Desktop/CAD/TP1/libwb/wbSolution.cpp.s
.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbSolution.cpp.s

home/the-user/Desktop/CAD/TP1/libwb/wbSparse.o: home/the-user/Desktop/CAD/TP1/libwb/wbSparse.cpp.o

.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbSparse.o

# target to build an object file
home/the-user/Desktop/CAD/TP1/libwb/wbSparse.cpp.o:
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/home/the-user/Desktop/CAD/TP1/libwb/wbSparse.cpp.o
.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbSparse.cpp.o

home/the-user/Desktop/CAD/TP1/libwb/wbSparse.i: home/the-user/Desktop/CAD/TP1/libwb/wbSparse.cpp.i

.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbSparse.i

# target to preprocess a source file
home/the-user/Desktop/CAD/TP1/libwb/wbSparse.cpp.i:
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/home/the-user/Desktop/CAD/TP1/libwb/wbSparse.cpp.i
.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbSparse.cpp.i

home/the-user/Desktop/CAD/TP1/libwb/wbSparse.s: home/the-user/Desktop/CAD/TP1/libwb/wbSparse.cpp.s

.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbSparse.s

# target to generate assembly for a file
home/the-user/Desktop/CAD/TP1/libwb/wbSparse.cpp.s:
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/home/the-user/Desktop/CAD/TP1/libwb/wbSparse.cpp.s
.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbSparse.cpp.s

home/the-user/Desktop/CAD/TP1/libwb/wbTimer.o: home/the-user/Desktop/CAD/TP1/libwb/wbTimer.cpp.o

.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbTimer.o

# target to build an object file
home/the-user/Desktop/CAD/TP1/libwb/wbTimer.cpp.o:
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/home/the-user/Desktop/CAD/TP1/libwb/wbTimer.cpp.o
.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbTimer.cpp.o

home/the-user/Desktop/CAD/TP1/libwb/wbTimer.i: home/the-user/Desktop/CAD/TP1/libwb/wbTimer.cpp.i

.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbTimer.i

# target to preprocess a source file
home/the-user/Desktop/CAD/TP1/libwb/wbTimer.cpp.i:
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/home/the-user/Desktop/CAD/TP1/libwb/wbTimer.cpp.i
.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbTimer.cpp.i

home/the-user/Desktop/CAD/TP1/libwb/wbTimer.s: home/the-user/Desktop/CAD/TP1/libwb/wbTimer.cpp.s

.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbTimer.s

# target to generate assembly for a file
home/the-user/Desktop/CAD/TP1/libwb/wbTimer.cpp.s:
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/home/the-user/Desktop/CAD/TP1/libwb/wbTimer.cpp.s
.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbTimer.cpp.s

home/the-user/Desktop/CAD/TP1/libwb/wbUtils.o: home/the-user/Desktop/CAD/TP1/libwb/wbUtils.cpp.o

.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbUtils.o

# target to build an object file
home/the-user/Desktop/CAD/TP1/libwb/wbUtils.cpp.o:
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/home/the-user/Desktop/CAD/TP1/libwb/wbUtils.cpp.o
.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbUtils.cpp.o

home/the-user/Desktop/CAD/TP1/libwb/wbUtils.i: home/the-user/Desktop/CAD/TP1/libwb/wbUtils.cpp.i

.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbUtils.i

# target to preprocess a source file
home/the-user/Desktop/CAD/TP1/libwb/wbUtils.cpp.i:
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/home/the-user/Desktop/CAD/TP1/libwb/wbUtils.cpp.i
.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbUtils.cpp.i

home/the-user/Desktop/CAD/TP1/libwb/wbUtils.s: home/the-user/Desktop/CAD/TP1/libwb/wbUtils.cpp.s

.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbUtils.s

# target to generate assembly for a file
home/the-user/Desktop/CAD/TP1/libwb/wbUtils.cpp.s:
	$(MAKE) -f CMakeFiles/wb.dir/build.make CMakeFiles/wb.dir/home/the-user/Desktop/CAD/TP1/libwb/wbUtils.cpp.s
.PHONY : home/the-user/Desktop/CAD/TP1/libwb/wbUtils.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... rebuild_cache"
	@echo "... wb"
	@echo "... edit_cache"
	@echo "... histogram_equalization"
	@echo "... clean_cuda_depends"
	@echo "... home/the-user/Desktop/CAD/TP1/libwb/vendor/json11.o"
	@echo "... home/the-user/Desktop/CAD/TP1/libwb/vendor/json11.i"
	@echo "... home/the-user/Desktop/CAD/TP1/libwb/vendor/json11.s"
	@echo "... home/the-user/Desktop/CAD/TP1/libwb/wbArg.o"
	@echo "... home/the-user/Desktop/CAD/TP1/libwb/wbArg.i"
	@echo "... home/the-user/Desktop/CAD/TP1/libwb/wbArg.s"
	@echo "... home/the-user/Desktop/CAD/TP1/libwb/wbCUDA.o"
	@echo "... home/the-user/Desktop/CAD/TP1/libwb/wbCUDA.i"
	@echo "... home/the-user/Desktop/CAD/TP1/libwb/wbCUDA.s"
	@echo "... home/the-user/Desktop/CAD/TP1/libwb/wbDataset.o"
	@echo "... home/the-user/Desktop/CAD/TP1/libwb/wbDataset.i"
	@echo "... home/the-user/Desktop/CAD/TP1/libwb/wbDataset.s"
	@echo "... home/the-user/Desktop/CAD/TP1/libwb/wbDirectory.o"
	@echo "... home/the-user/Desktop/CAD/TP1/libwb/wbDirectory.i"
	@echo "... home/the-user/Desktop/CAD/TP1/libwb/wbDirectory.s"
	@echo "... home/the-user/Desktop/CAD/TP1/libwb/wbExit.o"
	@echo "... home/the-user/Desktop/CAD/TP1/libwb/wbExit.i"
	@echo "... home/the-user/Desktop/CAD/TP1/libwb/wbExit.s"
	@echo "... home/the-user/Desktop/CAD/TP1/libwb/wbExport.o"
	@echo "... home/the-user/Desktop/CAD/TP1/libwb/wbExport.i"
	@echo "... home/the-user/Desktop/CAD/TP1/libwb/wbExport.s"
	@echo "... home/the-user/Desktop/CAD/TP1/libwb/wbFile.o"
	@echo "... home/the-user/Desktop/CAD/TP1/libwb/wbFile.i"
	@echo "... home/the-user/Desktop/CAD/TP1/libwb/wbFile.s"
	@echo "... home/the-user/Desktop/CAD/TP1/libwb/wbImage.o"
	@echo "... home/the-user/Desktop/CAD/TP1/libwb/wbImage.i"
	@echo "... home/the-user/Desktop/CAD/TP1/libwb/wbImage.s"
	@echo "... home/the-user/Desktop/CAD/TP1/libwb/wbImport.o"
	@echo "... home/the-user/Desktop/CAD/TP1/libwb/wbImport.i"
	@echo "... home/the-user/Desktop/CAD/TP1/libwb/wbImport.s"
	@echo "... home/the-user/Desktop/CAD/TP1/libwb/wbInit.o"
	@echo "... home/the-user/Desktop/CAD/TP1/libwb/wbInit.i"
	@echo "... home/the-user/Desktop/CAD/TP1/libwb/wbInit.s"
	@echo "... home/the-user/Desktop/CAD/TP1/libwb/wbLogger.o"
	@echo "... home/the-user/Desktop/CAD/TP1/libwb/wbLogger.i"
	@echo "... home/the-user/Desktop/CAD/TP1/libwb/wbLogger.s"
	@echo "... home/the-user/Desktop/CAD/TP1/libwb/wbMPI.o"
	@echo "... home/the-user/Desktop/CAD/TP1/libwb/wbMPI.i"
	@echo "... home/the-user/Desktop/CAD/TP1/libwb/wbMPI.s"
	@echo "... home/the-user/Desktop/CAD/TP1/libwb/wbPPM.o"
	@echo "... home/the-user/Desktop/CAD/TP1/libwb/wbPPM.i"
	@echo "... home/the-user/Desktop/CAD/TP1/libwb/wbPPM.s"
	@echo "... home/the-user/Desktop/CAD/TP1/libwb/wbPath.o"
	@echo "... home/the-user/Desktop/CAD/TP1/libwb/wbPath.i"
	@echo "... home/the-user/Desktop/CAD/TP1/libwb/wbPath.s"
	@echo "... home/the-user/Desktop/CAD/TP1/libwb/wbSolution.o"
	@echo "... home/the-user/Desktop/CAD/TP1/libwb/wbSolution.i"
	@echo "... home/the-user/Desktop/CAD/TP1/libwb/wbSolution.s"
	@echo "... home/the-user/Desktop/CAD/TP1/libwb/wbSparse.o"
	@echo "... home/the-user/Desktop/CAD/TP1/libwb/wbSparse.i"
	@echo "... home/the-user/Desktop/CAD/TP1/libwb/wbSparse.s"
	@echo "... home/the-user/Desktop/CAD/TP1/libwb/wbTimer.o"
	@echo "... home/the-user/Desktop/CAD/TP1/libwb/wbTimer.i"
	@echo "... home/the-user/Desktop/CAD/TP1/libwb/wbTimer.s"
	@echo "... home/the-user/Desktop/CAD/TP1/libwb/wbUtils.o"
	@echo "... home/the-user/Desktop/CAD/TP1/libwb/wbUtils.i"
	@echo "... home/the-user/Desktop/CAD/TP1/libwb/wbUtils.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

