package mnist.unsupervisedAutoEncoder;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;

public class MnistAutoencoder {

    public static void main(String[] args) throws Exception {
        DataSetIterator trainData = new MnistDataSetIterator(64, true, 12345);

        // Define the model configuration
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(12345)
            .updater(new Adam())
            .list()
            .layer(new DenseLayer.Builder().nIn(784).nOut(256)
                .activation(Activation.RELU).build())
            .layer(new DenseLayer.Builder().nIn(256).nOut(256)
                .activation(Activation.RELU).build())
            .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                .activation(Activation.SIGMOID).nIn(256).nOut(784).build())
            .build();

        // Initialize and train the model
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(5));

        int count =0;
        int epochs = 5;
        for (int i = 0; i < epochs; i++) {
            while (trainData.hasNext()) {
                DataSet batch = trainData.next();
                INDArray features = batch.getFeatures();
                model.fit(features, features); // Train with features as both input and output
                if( count % 200 == 0 )
                	visualizeResults(model, trainData);
                count++;
                
            }
        
            trainData.reset();
        }

        // Testing phase: visualize the first batch
        visualizeResults(model, trainData);
    }

    private static void visualizeResults(MultiLayerNetwork model, DataSetIterator data) throws Exception {
        JFrame frame = new JFrame("Reconstructed MNIST Images");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setLayout(new GridLayout(2, 10));

        if (data.hasNext()) {
            DataSet dataSet = data.next();
            INDArray features = dataSet.getFeatures();
            INDArray reconstructed = model.output(features);

            for (int i = 0; i < 10; i++) {
                frame.add(createImagePanel(features.getRow(i), "Original " + (i + 1)));
                frame.add(createImagePanel(reconstructed.getRow(i), "Reconstructed " + (i + 1)));
            }
        }

        frame.pack();
        frame.setVisible(true);
    }

    private static JPanel createImagePanel(INDArray imageRow, String title) {
        BufferedImage img = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
        for (int y = 0; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                int pixelVal = (int) (255 * imageRow.getDouble(y * 28 + x));
                int colorVal = (pixelVal << 16) | (pixelVal << 8) | pixelVal;
                img.setRGB(x, y, colorVal);
            }
        }

        JPanel panel = new JPanel();
        panel.setLayout(new BorderLayout());
        panel.add(new JLabel(title), BorderLayout.NORTH);
        panel.add(new JLabel(new ImageIcon(img)), BorderLayout.CENTER);
        return panel;
    }
}
