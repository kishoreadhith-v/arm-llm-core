#pragma once 

#include <iostream>
#include <fcntl.h>      
#include <sys/mman.h>   
#include <sys/stat.h>   
#include <unistd.h>     

#include "utils.h"

class ModelLoader {
private:
    int fd;             // To hold the File Descriptor
    size_t file_size;   // To hold the file size in bytes
    float* data;        // The raw memory pointer for our engine

public:
    // CONSTRUCTOR: Runs the moment you create the object
    ModelLoader(const char* filepath) {
        
        // ==========================================
        // TODO 1: Use open() to open 'filepath'. Store the result in 'fd'.
        fd = open(filepath, O_RDONLY);
        CHECK(fd != -1, "Failed to open weights file.");
        
        // TODO 2: Create a 'struct stat', use fstat() to read the file info, 
        //         and store the size in 'file_size'.
        struct stat filestats;
        CHECK(fstat(fd, &filestats) != -1, "Failed to read weights file stats.");
        file_size = filestats.st_size;
        

        // TODO 3: Use mmap() to map the file. 
        //         Because mmap returns a generic pointer (void*), you MUST 
        //         cast it to a float pointer like this: (float*)mmap(...)
        //         Store the result in 'data'.
        // ==========================================
        data = (float *) mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        CHECK(data != MAP_FAILED, "Failed to memory map file.");

        std::cout << "[*] Successfully mapped " << filepath << "\n";
    }

    // DESTRUCTOR: Runs automatically when the object is destroyed
    ~ModelLoader() {
        
        // ==========================================
        // TODO 4: Use munmap() to free the 'data' pointer.
        
        // TODO 5: Use close() to close the 'fd'.
        // ==========================================
        munmap(data, file_size);

        close(fd);
        std::cout << "Model Memory safely cleaned up.\n";
    }

    // A simple getter function so our main.cpp can access the pointer
    float* get_data() const {
        return data;
    }
};