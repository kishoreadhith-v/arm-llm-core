cmake_minimum_required(VERSION 3.15)

# Name your project and specify the language
project(ArmLLMCore CXX)

# We are using modern C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler Flags for Maximum Apple Silicon Performance
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mcpu=apple-m2 -Wall -Wextra")

# Tell the compiler where to look for header files (.h)
include_directories(include)

# Compile main.cpp into an executable called "llm_engine"
add_executable(llm_engine src/main.cpp)