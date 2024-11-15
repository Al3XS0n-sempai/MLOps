CXX=g++
CXXFLAGS=-std=c++11 -O3 -march=native -Wall -I$(SRC_DIR) $(shell python3 -m pybind11 --includes)
PY_LDFLAGS=$(shell python3-config --ldflags) -lopenblas -shared -fPIC
LDFLAGS=-lopenblas
SRC_DIR=relu_module/src
MODULE_DIR=relu_module/module

relu: $(MODULE_DIR)/py_relu_module.o $(SRC_DIR)/relu_bind.o
	$(CXX) $^ -o $(MODULE_DIR)/relu_binding`python3-config --extension-suffix` $(PY_LDFLAGS) $(CXXFLAGS)

$(MODULE_DIR)/py_relu_module.o: $(MODULE_DIR)/py_relu_module.cpp $(SRC_DIR)/relu_bind.hpp
	$(CXX) $(CXXFLAGS) -fPIC -c $< -o $@

$(SRC_DIR)/relu_bind.o: $(SRC_DIR)/relu_bind.cpp $(SRC_DIR)/relu_bind.hpp
	$(CXX) $(CXXFLAGS) -fPIC -c $< -o $@

clean:
	rm -f $(MODULE_DIR)/*.o $(SRC_DIR)/*.o $(MODULE_DIR)/linalg_core`python3-config --extension-suffix` $(TESTS_DIR)/test_linalg
