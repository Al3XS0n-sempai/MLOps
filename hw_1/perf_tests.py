# my ReLU
from relu_module import MyReLU

# torch-cpu ReLU
from torch import Tensor
from torch import nn

# tensorflow ReLU
import tensorflow as tf


def generate_inputs(count: int=1_000_000, seed: int=12345) -> list:
    res: list = []

    import numpy

    numpy.random.seed(seed)

    for _ in range(count):
        res.append(
            numpy.random.rand(
                numpy.random.randint(
                    100_000,
                    300_000
                )
            ) - 0.5
        )

    return res



def check_performance(func, inputs: list[any]) -> float:
    import cProfile

    test_code = '''
for test_input in inputs:
    _ = func(test_input)
'''

    cProfile.runctx(test_code, {'inputs': inputs, 'func': func}, {})




torch_relu = nn.ReLU()
tf_relu = tf.keras.activations.relu

inputs = generate_inputs(1_000)

check_performance(MyReLU, map(lambda x: x.tolist(), inputs))

check_performance(torch_relu, map(lambda x: Tensor(x), inputs))

check_performance(tf_relu, inputs)
