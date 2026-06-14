package JavaML_Mnist;

import JavaML_Core.*;

public class MnistParser implements DataParser<String> {
    private CoreConfig config;

    public MnistParser(CoreConfig config) {
        this.config = config;
    }

    @Override
    public DataPair parse(String line) throws Exception {
        String[] values = line.split(",");
        
        // Assuming the label is in the last column
        String labelValue = values[values.length - 1];
        
        // 1. One-Hot Encode the Label
        double[] labelData = new double[config.totalOutputFeatures];
        Integer labelIndex = config.labelMap.get(labelValue);
        if (labelIndex != null) {
            labelData[labelIndex] = 1.0;
        }
        Matrix labelMatrix = Matrix.create(labelData);

        // 2. Parse and Normalize Pixels (All indices except the last)
        double[] inputData = new double[config.totalInputFeatures];
        int featureIndex = 0;

        for (int i = 0; i < values.length - 1; i++) {
            if (config.isNumeric[i]) {
                double rawPixel = Double.parseDouble(values[i]);
                double mean = config.columnMeans.get(i);
                double std = config.columnStds.get(i);
                
                // Standardize the pixel: (x - mean) / std
                inputData[featureIndex++] = (std == 0) ? 0 : (rawPixel - mean) / std;
            }
        }

        Matrix inputMatrix = Matrix.create(inputData);

        return new DataPair(inputMatrix, labelMatrix);
    }
}