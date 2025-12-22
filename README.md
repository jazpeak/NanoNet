# NanoNet

A lightweight, runtime C++ inference engine for running simple ONNX models.

## Future Goals
I plan to extend support to any onnx/ML/DL model in version 2 of this project. In Version 3, i plan to make it into a full training framework, similar to Pytorch.
Further, Several optimizations are possible including using CUDA, which is a distant Goal once i get time and the knowledge in several domains. Thanks for reading this!!!

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
   **Important:** Run these scripts from inside the `build` directory so the generated files are placed where the engine expects them.

   ```bash
   cd build
   
   # 1. Train the model
   # Trains a simple Multi-Layer Perceptron (MLP) on the MNIST dataset using PyTorch.
   # - Architecture: Flatten -> Linear(784, 128) -> ReLU -> Linear(128, 10)
   # - Saves weights to 'mnist_mlp.pt'
   python3 ../mnist_scripts/train.py 
   
   # 2. Export to ONNX
   # Loads the trained PyTorch model and exports it to the standard ONNX format.
   # - Generates 'mnist_mlp.onnx' (Defines the computation graph for our C++ engine)
   python3 ../mnist_scripts/export.py 
   
   # 3. Generate Verification Data
   # Runs inference using PyTorch on the first 100 test images and dumps the verification data:
   # - 'mnist_X.txt': Input image tensors (normalized pixel values 28x28 flattened)
   # - 'mnist_Y.txt': Ground truth class labels (0-9)
   # - 'mnist_logits_ref.txt': The raw output logits computed by PyTorch (Reference for numerical accuracy)
   python3 ../mnist_scripts/dump.py 
   ```
3. **Run the inference engine**:
   The `test_mnist` executable loads the ONNX model and the verification data. It runs the C++ inference engine on the inputs and compares the results:
   - **Numerical Check**: Compares C++ output logits vs PyTorch reference logits (ensures math is identical).
   - **Accuracy Check**: Compares predicted class vs ground truth label.
   
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

## Note
LLMs were used to generate the documentation and README.md file. It was also used to partially generate the train.py, export.py, dump.py, and test_mnist.cpp files. The rest was done by me.
