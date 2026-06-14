package JavaML_Mnist;

import JavaML_Core.*;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.HashSet;
import java.util.Set;

public class MnistDriver {

    public static void main(String[] args) throws Exception {
        MnistConfig config = new MnistConfig();

        System.out.println("--- MNIST Classification Engine ---");

        // 1. Ensure reproducibility and split the raw data before anything else happens
        DataSplitter.generateSplitsIfMissing(config);

        System.out.println("Initiating one-time metadata scan...");
        Set<Integer> ignoreCols = new HashSet<>();
        int labelCol = 784; // The Kaggle target variable index (131st comma)

        // Scan schema to auto-configure normalization and output mappings
        config.scanDataSchema(new String[] { config.rawSourcePath }, ignoreCols, labelCol);

        // Populate the dynamic architecture layers (input & output)
        config.nodes[0] = config.totalInputFeatures;
        config.nodes[config.nodes.length - 1] = config.totalOutputFeatures;

        // 2. Inject the globally scanned schema into the specific parser
        MnistParser parser = new MnistParser(config);

        // 3. Inject the parser into the file streaming engines
        StreamingIterator[] iterators = loadData(config, parser);

        NeuralNetwork model;
        if (new File(config.loadModelPath).exists()) {
            System.out.println("Loading neural network architecture...");
            model = NeuralNetwork.load(config.loadModelPath);
        } else {
            System.out.println("Creating neural network architecture...");
            model = NeuralNetwork.createModel(config.nodes, config.activations, config.costFunction);
        }
        model.printStructure();

        // 4. Execute the pipeline
        trainModel(model, iterators[0], iterators[1], config);
        evaluatePredictions(model, iterators[2], config);
    }

    private static StreamingIterator[] loadData(MnistConfig config, MnistParser parser) throws Exception {
        System.out.println("Loading data streams...");
        StreamingIterator trainer = new StreamingIterator(config.batchSize, config.trainPath, config.hasHeader, parser);
        StreamingIterator validator = new StreamingIterator(config.batchSize, config.validationPath, config.hasHeader,
                parser);
        StreamingIterator tester = new StreamingIterator(config.batchSize, config.testPath, config.hasHeader, parser);
        return new StreamingIterator[] { trainer, validator, tester };
    }

    private static void trainModel(NeuralNetwork model, StreamingIterator trainer, StreamingIterator validator,
            MnistConfig config) throws Exception {
        System.out.println("\nInitializing training pipeline...");
        System.out.println("Batch Size: " + config.batchSize + " | Epochs: " + config.epochs + " | Decay Rate: "
                + config.decayRate);

        // 1. Calculate the baseline accuracy before any training happens
        double bestValAccuracy = model.getClassifierAccuracy(validator);

        System.out.printf("Baseline Validation Accuracy (Epoch 0): %.2f%%\n\n", bestValAccuracy);

        double currentLearningRate = config.learningRate;
        long totalStartTime = System.currentTimeMillis();

        // 2. Training loop
        for (int epoch = 1; epoch <= config.epochs; epoch++) {
            long epochStartTime = System.currentTimeMillis();

            // Capture the statistics returned from the training run
            int[] trainingStats = model.train(trainer, currentLearningRate);
            int dataPointsTrained = trainingStats[1];

            double currentAccuracy = model.getClassifierAccuracy(validator);

            // For classification, higher accuracy is better
            if (currentAccuracy > bestValAccuracy) {
                System.out.println(">>> New best model found! Saving weights to " + config.saveModelPath);
                bestValAccuracy = currentAccuracy;
                model.save(config.saveModelPath);
            }

            double epochTimeSec = (System.currentTimeMillis() - epochStartTime) / 1000.0;

            System.out.printf("[Epoch %d | %5.2fs] LR: %.5f | Rows: %,d | Val Accuracy: %.2f%%\n",
                    epoch, epochTimeSec, currentLearningRate, dataPointsTrained, currentAccuracy);

            currentLearningRate *= config.decayRate;
        }

        double totalTimeMin = (System.currentTimeMillis() - totalStartTime) / 60000.0;
        System.out.printf("\nPipeline Complete in %.2f minutes!\n", totalTimeMin);
    }

    private static void evaluatePredictions(NeuralNetwork model, StreamingIterator tester, MnistConfig config)
            throws Exception {
        System.out.println("\n--- Out-of-Sample Predictions ---");
        double finalAccuracy = model.getClassifierAccuracy(tester);
        System.out.printf("Final Test Set Accuracy: %.2f%%\n\n", finalAccuracy);

        DataPair[] dataPairs = tester.get(0, 15);
        ImageViewer viewer = new ImageViewer("MNIST Prediction Showcase");
        viewer.show();

        for (DataPair pair : dataPairs) {
            int expectedDigit = pair.label.maxIndex();
            int predictedDigit = model.compute(pair.input).maxIndex();

            Matrix displayMatrix = createDisplayMatrix(pair.input, config);
            viewer.draw(displayMatrix, 10.0);

            System.out.printf("Expected: %d | Predicted: %d %s\n",
                    expectedDigit, predictedDigit, (expectedDigit == predictedDigit ? "✓" : "[WRONG]"));
            System.out.println("---------------------------");

            // Pause longer on incorrect predictions so you can inspect them
            Thread.sleep(expectedDigit == predictedDigit ? 1000 : 2000);
        }

        viewer.hide();
        System.exit(0); // Ensures the Swing GUI thread terminates completely
    }

    private static Matrix createDisplayMatrix(Matrix normalizedInput, MnistConfig config) throws Exception {
        Matrix displayMatrix = Matrix.create(28, 28);
        double[] cells = normalizedInput.getCells();

        int featureIndex = 0;
        // Iterate up to the last column (which is the label)
        for (int i = 0; i < config.isNumeric.length - 1; i++) {
            if (config.isNumeric[i]) {
                double std = config.columnStds.get(i);
                double mean = config.columnMeans.get(i);

                // Reverse the standardization to get raw pixel values
                double rawPixel = (cells[featureIndex] * (std == 0 ? 1.0 : std)) + mean;

                // Clamp to valid RGB bounds (0-255)
                rawPixel = Math.max(0, Math.min(255, rawPixel));

                displayMatrix.set(featureIndex / 28, featureIndex % 28, rawPixel);
                featureIndex++;
            }
        }
        return displayMatrix;
    }
}