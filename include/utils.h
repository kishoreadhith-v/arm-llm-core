#pragma once

#include <iostream>
#include <cstdlib>
#include <cerrno>
#include <cstring>

const std::string red("\033[31m");
const std::string green("\033[32m");
const std::string yellow("\033[33m");
const std::string blue("\033[34m");
const std::string reset("\033[0m"); // Reset to default color

#define CHECK(condition, message)                                                                                             \
    do                                                                                                                        \
    {                                                                                                                         \
        if (!(condition))                                                                                                     \
        {                                                                                                                     \
            std::cout << red << "------------------------------------------------------------" << reset << "\n"; \
            std::cout << red << "ERROR: " << reset << message << "\n";                                           \
            std::cout << __FILE__ << " : " << __LINE__ << "\n";                                                               \
            std::cout << strerror(errno) << "\n";                                                                             \
            std::cout << red << "------------------------------------------------------------" << reset << "\n"; \
            exit(1);                                                                                                          \
        }                                                                                                                     \
    } while (0)