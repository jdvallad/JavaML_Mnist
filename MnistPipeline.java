package JavaML_Mnist;

import JavaML_Core.ClassificationPipeline;
import JavaML_Core.CoreConfig;
import JavaML_Core.DataPair;
import JavaML_Core.ImageViewer;
import JavaML_Core.Matrix;
import JavaML_Core.NeuralNetwork;
import JavaML_Core.StreamingIterator;

public class MnistPipeline extends ClassificationPipeline {

    public MnistPipeline(CoreConfig config) {
        super(config);
    }

    @Override
    protected void evaluatePredictions(NeuralNetwork model, StreamingIterator tester) throws Exception {
        System.out.println("\n--- Out-of-Sample Predictions ---");
        DataPair[] pairs = tester.get(0, config.displayCount);

        ImageViewer viewer = new ImageViewer("MNIST Predictions");

        String[] labels = new String[config.totalOutputFeatures];
        config.labelMap.forEach((k, v) -> labels[v] = k);

        for (DataPair p : pairs) {
            double[] expectedArr = p.label.getCells();
            postProcessOutput(expectedArr);
            int actual = 0;
            for (int i = 1; i < expectedArr.length; i++) {
                if (expectedArr[i] > expectedArr[actual])
                    actual = i;
            }

            double[] predictedArr = model.compute(p.input).getCells();
            postProcessOutput(predictedArr);
            int pred = 0;
            for (int i = 1; i < predictedArr.length; i++) {
                if (predictedArr[i] > predictedArr[pred])
                    pred = i;
            }

            System.out.printf("Expected: %s | Predicted: %s %s\n", labels[actual], labels[pred],
                    (pred == actual ? "✓" : "✗"));

            try {
                // Reconstruct the 28x28 image and display it
                // Un-normalize the pixel features so they render correctly (0-255)
                Matrix imageMatrix = p.input.clone();
                for (int i = 0; i < imageMatrix.getColumns(); i++) {
                    double std = config.columnStds.getOrDefault(i, 1.0);
                    double mean = config.columnMeans.getOrDefault(i, 0.0);
                    imageMatrix.set(0, i, (imageMatrix.get(0, i) * std) + mean);
                }
                viewer.draw(imageMatrix.shape(28, 28), 10.0);
                viewer.show();
            } catch (Exception e) {
            }

            Thread.sleep((pred == actual) ? 500 : 2000);
        }
        viewer.hide();
    }

    public static void main(String[] args) throws Exception {
        System.out.println("--- MNIST Classification Engine ---");
        MnistConfig config = new MnistConfig();
        MnistPipeline pipeline = new MnistPipeline(config);
        pipeline.execute();
        System.exit(0);
    }
}