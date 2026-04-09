#!/bin/bash

# 1. Create the build directory if it doesn't exist yet (-p prevents errors if it does)
mkdir -p build

# 2. Step inside the build directory
cd build

# 3. Run CMake to generate the Makefiles (looking one folder up '..')
echo "[*] Generating CMake Build System..."
cmake ..

# 4. Compile the C++ code
echo "[*] Compiling Engine..."
make

# 5. Run the Engine automatically if compilation was successful
echo "[*] Build Complete. Executing..."
echo ""
./llm_engine