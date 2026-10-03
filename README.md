# Forward-Forward

A neural network implemented from scratch in modern C++, with a terminal demo that learns whether a digit image matches a supplied label.

The hidden layers learn through local Forward-Forward goodness updates. A separately trained sigmoid output produces a score for the image-label pair. The project uses custom vector and matrix classes and the C++ standard library.

## Build and run

The command below has been used successfully on Linux with GCC 16. You need `g++-16`, a matching standard library with `std::format` support, and a terminal that supports ANSI cursor movement and colors. No external ML libraries or dataset downloads are required.

Clone the repository:

```bash
git clone https://github.com/cartercpp/forward-forward.git
cd forward-forward
```

Compile and launch:

```bash
g++-16 -std=c++26 -O3 -march=native -flto -ffast-math -pthread \
  '-Dpre(...)=' '-Dpost(...)=' \
  main.cpp forward_forward.cpp -o main && ./main
```

Press **Enter** to stop training and exit. To run the compiled program again:

```bash
./main
```

### Compiler options

| Option | Purpose |
| --- | --- |
| `-std=c++26` | Selects C++26 mode. |
| `-O3` | Enables aggressive optimization. |
| `-march=native` | Optimizes for the CPU on the build machine. Rebuild on another machine if its CPU differs. |
| `-flto` | Enables link-time optimization across source files. |
| `-ffast-math` | Allows more aggressive floating-point optimization, which can speed up dot products. It can change numerical results and NaN/infinity behavior. |
| `-pthread` | Enables the compiler and linker support needed for threading. |
| `'-Dpre(...)=' '-Dpost(...)='` | Removes contract annotations during preprocessing to work around GCC contract compilation crashes. Contract checks are disabled in this build. |

Keep the quotes around the two macro definitions. This build deliberately omits `-fcontracts`. To compare training without fast-math optimizations, remove only `-ffast-math`.

## What the demo does

`main.cpp` contains ten hand-coded **28 x 28 binary images**, one for each digit from 0 to 9.

1. A digit image supplies 784 pixel values.
2. Ten more input values encode a candidate label as a one-hot vector: the candidate digit's position is 1 and the other positions are 0.
3. An image paired with its correct label is trained with `fit(input, true)`.
4. An image of a different digit paired with that label is trained with `fit(input, false)`.
5. The terminal displays a training image and the sigmoid score for its correct label, formatted as a percentage.

The display refreshes every 1,000 loop iterations, including the first iteration, and pauses for 500 milliseconds after each refresh. Training runs in a `std::jthread` until you press Enter.

These are built-in examples; the program does not load MNIST. The displayed percentage is a model score on a training example, not measured test-set accuracy.

## Learning rule

Each hidden layer uses ReLU activations. Its goodness is the **mean of the squared activations**:

```text
goodness = sum(activation * activation) / number_of_neurons
```

For a positive example, a layer's weights and biases are adjusted to increase goodness when it is below the threshold. For a negative example, they are adjusted to decrease goodness when it is above the threshold.

The final sigmoid layer learns the same positive/negative target with its own local update. Its error is not backpropagated through the hidden layers.

`predict(input)` returns a single image-label match score. To choose among digits 0-9, evaluate the same image with each candidate label and select the highest score; the current demo displays the supplied correct label's score.

## Configuration

The default network in `main.cpp` is:

```cpp
forward_forward digitRecognizer({pixels + labels, 100, 100, 1}, 0.01, 2);
```

| Setting | Default |
| --- | --- |
| Input size | 794: 784 pixels + 10 label values |
| Hidden layers | 100 neurons each, with ReLU |
| Output | 1 sigmoid neuron |
| Learning rate | 0.01 |
| Goodness threshold | 2 |

The training loop uses a reusable stack buffer and a per-iteration `std::pmr::monotonic_buffer_resource` for the input vector. The network uses its default memory resource unless a resource is supplied to its constructor.

## Files

| File | Purpose |
| --- | --- |
| `main.cpp` | Digit patterns, positive/negative training examples, and terminal visualization. |
| `forward_forward.hpp` / `forward_forward.cpp` | Network construction, local goodness training, and prediction. |
| `math_vector.hpp` | Vector arithmetic and dot products. |
| `matrix.hpp` | Matrix arithmetic, matrix-vector multiplication, and outer products. |
| `CMakeLists.txt` | CMake project definition. The direct compiler command above includes the contract workaround and optimization flags. |
