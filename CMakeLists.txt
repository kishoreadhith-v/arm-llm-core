cmake_minimum_required(VERSION 3.15)

# Name your project and specify the language
project(ArmLLMCore CXX)

# We are using modern C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler Flags for Maximum Apple Silicon Performance
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mcpu=apple-m2 -ffast-math -Wall -Wextra")

# Tell the compiler where to look for header files (.h)
include_directories(include)

# Compile main.cpp into an executable called "llm_engine"
add_executable(llm_engine src/main.cpp)

# find and link open-mp for multi-threading
if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    # force the preprocessor to accept the flag
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
    
    # library file path
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L/opt/homebrew/opt/libomp/lib -lomp")
else()
    # Standard fallback for Linux/Windows
    find_package(OpenMP REQUIRED)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(llm_engine PUBLIC OpenMP::OpenMP_CXX)
    endif()
endif()