# --- setting up system tools ---

FROM python:3.10-slim AS system

RUN apt-get update
RUN apt-get install -y g++ libopenblas-dev build-essential

RUN pip install pybind11

# --- compiling module ---

FROM system AS compiler

WORKDIR /module

COPY ./relu_module/ /module/relu_module
COPY ./Makefile /module/Makefile

RUN make relu # make rule inside /module

# --- setuptools ---

FROM compiler AS setuptools

COPY ./setup.py /module/setup.py
COPY ./pyproject.toml /module/pyproject.toml

RUN pip install --no-cache setuptools build

RUN python3 -m build
RUN pip install /module/dist/MyReLU-*.whl

# --- RUNNING PYTESTS ---

FROM setuptools AS pytests

COPY ./tests.py /module/tests.py

RUN pip install pytest
RUN pytest /module/tests.py

# --- RUNNING PREFORMANCE TESTS ---

FROM pytests AS perftests

COPY ./perf_tests.py /module/perf_tests.py

RUN pip install --no-cache-dir tensorflow
RUN pip install --no-cache-dir torch==2.5.0 --index-url https://download.pytorch.org/whl/cpu

CMD ["python", "/module/perf_tests.py"]

