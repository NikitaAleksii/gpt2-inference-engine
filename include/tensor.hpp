#include <vector>
#include <memory>
#include <algorithm>

class Tensor {
private:
    std::vector<int> shape;                     // shape of the tensor
    float* buffer;                              // pointer to the tensor
    std::vector<int> strides;                   // number of array positions you must skip in physical memory to move one step forward along a specific dimension
    bool owns_buffer;                           // true - activation; false - points to weights

    // Computes strides
    void compute_strides() {
        strides.resize(shape.size());
        size_t acc = 1;
        for (int i = shape.size() - 1; i >= 0; i--) {
            strides[i] = acc;
            acc *= shape[i];
        }
    }

    // Computes the size of a flat array for N-dimensional tensor
    size_t compute_size() {
        size_t size = 1;
        for (int i = 0; i < this->shape.size(); i++) {
            size *= this->shape[i];
        }
        return size;
    }

    // Row-major mapping to index 
    size_t offset(const std::vector<int>& idx) const {
        size_t offset = 0;
        for (int i = 0; i < idx.size(); i++) {
            offset += this->strides[i] * idx[i];
        }
        return offset;
    }

public:
    // General constructor
    Tensor() {
        this->buffer      = nullptr;
        compute_strides();
        this->owns_buffer = false;
    };

    // Constructor to allocate fresh memory for activations 
    Tensor(std::vector<int>& shape); 

    // Constructor to used already allocated memory for weights
    Tensor(std::vector<int>& shape, void* data, size_t byte_offset);

    // Destructor
    ~Tensor() {
        if (owns_buffer) 
            delete[] buffer;
    }

    // Copy constructor
    Tensor(const Tensor& other) {
        // Copy `shape`, `stride`, and `owns_buffer` to a new tensor
        this->shape       = other.shape;
        compute_strides();
        this->owns_buffer = true;

        // Create a new buffer and copy contents from another buffer
        int n        = compute_size();
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
        compute_strides();
        this->owns_buffer = true;

        // Create a new buffer and copy contents from another buffer
        int n        = compute_size();
        this->buffer = new float[n];
        std::copy(other.buffer, other.buffer + n, this->buffer);

        return *this;
    }

    // Move assignment
    Tensor(Tensor&& other) noexcept {
        // Copy `shape`, `stride`, and `ows_buffer` to a new tensor
        this->shape       = other.shape;
        compute_strides();
        this->owns_buffer = other.owns_buffer;

        // Steal the pointer
        this->buffer      = other.buffer;

        // Empty previous pointer
        other.buffer      = nullptr;
        other.owns_buffer = false;
    }

    // Move assignment operator
    Tensor& operator=(Tensor&& other) noexcept {
        // Guard against self-moving
        if (this == &other) {
            return *this;
        }

        // First check if we can delete data in memory and then assign `owns_buffer` to true
        if (owns_buffer) {
            delete[] buffer;
        }

        // Copy `shape`, `stride`, and `ows_buffer` to a new tensor
        this->shape       = other.shape;
        compute_strides();
        this->owns_buffer = other.owns_buffer;

        // Steal the pointer
        this->buffer      = other.buffer;

        // Empty previous pointer
        other.buffer      = nullptr;
        other.owns_buffer = false;

        return *this;
    }

    // Returns an element stored in a flat tensor
    float& at(const std::vector<int>& idx) {
        return this->buffer[offset(idx)];
    }

};
