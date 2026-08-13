#include "loader.hpp"

Loader::Loader(const char* filepath) {
    // Open file descriptor
    int fd = open(filepath, O_RDONLY);
    if (fd == -1) {
        throw std::runtime_error(std::string("Error opening file: ") + filepath);
    }

    // Get the file size
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        close(fd); 
        throw std::runtime_error(std::string("Error getting file size"));
    }
    this->length = sb.st_size;

    // Map the file into memory
    this->addr = static_cast<char*>(mmap(nullptr, this->length, PROT_READ, MAP_PRIVATE, fd, 0));
    close(fd);
    if (this->addr == MAP_FAILED) {
        throw std::runtime_error(std::string("Error mapping file"));
    }

    // Read header of the file
    uint64_t header_len;
    std::memcpy(&header_len, addr, 8);
    std::string header(addr + 8, header_len);

    // Parse JSON and save each entry into the weight map
    json data = json::parse(header);
    for (auto it = data.begin(); it != data.end(); ++it) {
        if (it.key() == "__metadata__")
            continue;

        auto data_offsets = (*it)["data_offsets"];          // offset in format [begin, end]
        std::vector<int> shape = (*it)["shape"];
        size_t byte_offset  = 8 + header_len + static_cast<size_t>(data_offsets[0]);

        weight_map[it.key()] = Tensor(shape, addr, byte_offset);
    }
} 

int main() {
    Loader load("../models/model.safetensors");
    std::cout << load.weight_map.size();
    return 0;
}