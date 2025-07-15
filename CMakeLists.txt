cmake_minimum_required(VERSION 3.10)
project(VectorPruningExperiment CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 找到所有源文件
file(GLOB_RECURSE SOURCES "src/*.h" "src/*.cpp")

# 添加可执行文件
add_executable(pruning_experiment ${SOURCES})

# 推荐：在开发时使用Debug模式，发布时使用Release模式以获得最佳性能
# 可以通过 cmake .. -DCMAKE_BUILD_TYPE=Release 来指定
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
