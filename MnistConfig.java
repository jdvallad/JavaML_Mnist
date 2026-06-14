package JavaML_Mnist;

import JavaML_Core.CoreConfig;

public class MnistConfig extends CoreConfig {
    public MnistConfig() {
        // MNIST is a classification problem (digits 0-9)
        this.isClassification = true;
        this.hasHeader = true; // Most curled CSVs include a header row

        // Hyperparameters
        this.batchSize = 64;
        this.learningRate = 0.01;
        this.epochs = 5;

        // File Paths
        // Note: Check if your CSV folder is also lowercase 'mnist' or uppercase 'Mnist'
        this.rawSourcePath = "./JavaML_Mnist/csv/mnist.csv";
        this.trainPath = "./JavaML_Mnist/csv/mnist_train.csv";
        this.validationPath = "./JavaML_Mnist/csv/mnist_val.csv";
        this.testPath = "./JavaML_Mnist/csv/mnist_test.csv";

        this.saveModelPath = "./JavaML_Mnist/models/mnist_model.ser";
        this.loadModelPath = "./JavaML_Mnist/models/mnist_model.ser";

        // Architecture: 784 inputs -> 128 hidden -> 10 outputs
        // (0s are placeholders updated by scanDataSchema)
        this.nodes = new int[] { 0, 128, 0 };

        // Classification usually pairs well with Softmax on the output
        this.activations = new String[] { "leakyRelu", "softmax" };
        this.costFunction = "logLoss";
    }
}