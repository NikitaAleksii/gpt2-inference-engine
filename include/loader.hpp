#pragma once

#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <stdexcept>
#include <cstdint>
#include <unordered_map>
#include <json.hpp>
#include "tensor.hpp"

using json = nlohmann::json;

class Loader {
private:
    char* addr    = nullptr;
    size_t length = 0;

public:
    // Map
    std::unordered_map<std::string, Tensor> weight_map;

    // Constructor that populates `addr` and `length`
    Loader(const char* filepath);

    // Destructor
    ~Loader() {
        if (addr != MAP_FAILED && addr != nullptr) {
            munmap(addr, length);
        }
    }

    // Copy constructor is not needed
    Loader(const Loader& other) = delete;
    
    // Copy constructor operator is not needed
    Loader& operator=(const Loader& other) = delete;
};