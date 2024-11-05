#include "relu_bind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


namespace py = pybind11;

PYBIND11_MODULE(relu_binding, m) {
    m.doc() = "My module with relu function";

    m.def(
        "my_relu",
        &ReLU::ReLU,
        "Calculating ReLU of the vector"
    );
}
