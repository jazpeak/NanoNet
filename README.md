# NanoNet

A lightweight, dependency-free (runtime) C++ inference engine for running simple ONNX models.

## Features

- **Zero Runtime Dependencies**: Only requires a C++17 compiler and standard libraries (Protobuf/ONNX used only for model loading).
- **Parallel Execution**: Multi-threaded Matrix Multiplication (GEMM) implementation.
- **ONNX Support**: Loads and executes standard ONNX models with embedded or external data.
- **Implemented Operators**:
  - `Gemm` (General Matrix Multiply) with Transpose B support
  - `Relu`
  - `Flatten`
  - `Reshape`
  - `Add` (Bias addition)
- **Extensible Graph API**: Simple node-based graph execution engine.

## Prerequisites

- **CMake** (3.10+)
- **C++ Compiler** (supporting C++17)
- **Protobuf Compiler** (`protoc`) and libraries (required for building the ONNX loader)
- **Python 3** (optional, for generating test models) - requires `torch`, `torchvision`, `onnx`

## Building

```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
```

## Running MNIST inference

The project includes an end-to-end test with the MNIST dataset.

1. **Build the project** (see above).
2. **Generate the model and data**:
   Inside the `build` directory, run the Python scripts to train a simple PyTorch model, export it to ONNX, and dump test data.
   ```bash
   # Generates mnist_mlp.pt and data folder
   python3 ../mnist_scripts/train.py 
   
   # Exports to mnist_mlp.onnx and verifies correctness
   python3 ../mnist_scripts/export.py 
   
   # Dumps input/output vectors for C++ verification
   python3 ../mnist_scripts/dump.py 
   ```
3. **Run the inference engine**:
   ```bash
   ./test_mnist
   ```
   
   Expected output:
   ```
   === Single image ===
   Max logit error: <very small number>
   Pred: 7  True: 7

   === Batch test ===
   Accuracy over 100: 98%
   ```

## Project Structure

- `include/`: Header files defining the core API (`Tensor`, `Graph`, `Node`).
- `src/`: Implementation of operators, tensor logic, and the ONNX loader.
- `onnx/`: ONNX submodule (protobuf definitions).
- `tests/`: specific accuracy tests.

## Note
This is an educational project aimed at understanding the internals of deep learning inference engines. It is not optimized for production use on large models.