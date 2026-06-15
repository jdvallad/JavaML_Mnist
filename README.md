# JavaML - MNIST Classification

This repository contains the MNIST handwritten digit classification implementation, built on top of the custom `JavaML_Core` neural network engine.

## Prerequisites

- Java Development Kit (JDK)
- `wget` (to download the dataset)
- The `JavaML_Core` engine (must be cloned alongside this repository)

## Data Setup

Before running the model, you need to download the raw MNIST dataset. Run the following command from within this directory (`JavaML_Mnist`) to fetch the dataset and place it in the correct folder:

```bash
mkdir -p ./data && wget -c --tries=1000 --retry-connrefused -O ./data/mnist.csv "https://www.openml.org/data/get_csv/52667/mnist_784.arff"
```

*(Note: The `MnistConfig` expects this data to exist at `./JavaML_Mnist/data/mnist.csv` relative to the parent execution directory).*

## Project Structure

This module relies on the pure-Java ML engine located in the `JavaML_Core` repository. To run this project, both repositories should be cloned side-by-side in the same parent directory:

```text
/your-folder/
├── JavaML_Core/
└── JavaML_Mnist/
```

## Compiling and Running

1. Open your terminal and navigate to the parent folder containing both repositories.
2. Compile the core engine alongside this module:
   ```bash
   javac JavaML_Core/*.java JavaML_Mnist/*.java
   ```
3. Run the driver:
   ```bash
   java JavaML_Mnist.MnistDriver
   ```

The driver will automatically split the raw data, train the neural network, and launch a Java Swing GUI to visualize the final out-of-sample predictions!

## Expected Results

Out of the box, this architecture (784 Input -> 128 Hidden -> 10 Output nodes) trains for 10 epochs using Log-Loss and Softmax activation. 

Because the entire neural network engine is built from scratch in pure Java (featuring custom cache-optimized matrix multiplications), the pipeline streams and trains extremely quickly, typically achieving **>95% out-of-sample accuracy** on the test set.