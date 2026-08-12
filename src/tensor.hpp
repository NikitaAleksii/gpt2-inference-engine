#include <vector>
#include <memory>
#include <algorithm>

class Tensor {
private:
    std::vector<int> shape;                     // shape of the tensor
    float* buffer;                              // pointer to the tensor
    int stride;                                 // how many steps to get to another row
    bool owns_buffer;                           // true - activation; false - points to weights

public:
    // General constructor
    Tensor() {
        this->buffer      = nullptr;
        this->stride      = 0;
        this->owns_buffer = false;
    };

    // Constructor to allocate fresh memory for activations 
    Tensor(std::vector<int>& shape); 

    // Constructor to used already allocated memory for weights
    Tensor(std::vector<int>& shape, void* data, size_t byte_offset);

    // Row-major mapping to index 
    int index(int row, int col) {
        return this->stride * row + col;
    }

    // Returns an element stored in a flat tensor
    float& at(int row, int col) {
        return this->buffer[index(row, col)];
    }

    // Destructor
    ~Tensor() {
        if (owns_buffer) 
            delete[] buffer;
    }

    // Copy constructor
    Tensor(const Tensor& other) {
        // Copy `shape`, `stride`, and `ows_buffer` to a new tensor
        this->shape       = other.shape;
        this->stride      = other.stride;
        this->owns_buffer = true;

        // Create a new buffer and copy contents from another buffer
        int n        = other.shape[0] * other.shape[1];
        this->buffer = new float[n];
        std::copy(other.buffer, other.buffer + n, this->buffer);
    }

    // Copy assignment operator
    Tensor& operator=(const Tensor& other) {
        // Guard against self-assignment
        if (this == &other) {
            return *this;
        }

        // First check if we can delete data in memory and then assign `owns_buffer` to true
        if (owns_buffer) {
            delete[] buffer;
        }

        // Copy `shape`, `stride`, and `ows_buffer` to a new tensor
        this->shape       = other.shape;
        this->stride      = other.stride;
        this->owns_buffer = true;

        // Create a new buffer and copy contents from another buffer
        int n        = other.shape[0] * other.shape[1];
        this->buffer = new float[n];
        std::copy(other.buffer, other.buffer + n, this->buffer);

        return *this;
    }
};