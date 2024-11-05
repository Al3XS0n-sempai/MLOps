#include "relu_bind.hpp"

#include <cstdlib>
#include <vector>


using std::vector;
using std::size_t;

vector<double> ReLU::ReLU(const vector<double>& vec) {
    vector<double> new_vec(vec.size());

    for (size_t i = 0; i < vec.size(); ++i) {
        new_vec[i] = ((vec[i] < 0.) ? 0. : vec[i]);
    }

    return new_vec;
}
