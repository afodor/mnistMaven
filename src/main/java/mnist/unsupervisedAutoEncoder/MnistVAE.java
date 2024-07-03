package mnist.unsupervisedAutoEncoder;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.ops.transforms.Transforms;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.Map;

public class MnistVAE {

    public static void main(String[] args) throws Exception {
        int batchSize = 64;
        MnistDataSetIterator trainData = new MnistDataSetIterator(batchSize, true, 12345);

        ComputationGraph vae = createVAE();
        vae.init();

        // Sample from latent space and visualize
        INDArray sampledImages = sampleFromLatentSpace(vae, 10,784);
        visualize(sampledImages);

        // Train model
        
        int epochs = 10;
        for (int i = 0; i < epochs; i++) {
            trainModel(vae, trainData);
            trainData.reset();

            // Sample from latent space and visualize
            sampledImages = sampleFromLatentSpace(vae, 10,784);
            visualize(sampledImages);

        }
    }
    
    public static INDArray sampleFromLatentSpace(ComputationGraph vae, int numSamples, int latentDim) {
        // Generate random points in the latent space
        INDArray randomLatentSpace = Nd4j.randn(numSamples, latentDim);

        // Running the decoder part from the latent space
        // Assuming "decoder_start" is the name of the first decoder layer in your graph
        INDArray output = vae.outputSingle(false, randomLatentSpace);

        return output;
    }


    public static ComputationGraph createVAE() {
        int imageSize = 784; // 28*28
        int latentDim = 2; // 2D latent space

        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(12345)
            .updater(new Adam(0.001))
            .graphBuilder()
            .addInputs("input")
            .addLayer("encoder_hidden", new DenseLayer.Builder().nIn(imageSize).nOut(256).activation(Activation.RELU).build(), "input")
            .addLayer("mean", new DenseLayer.Builder().nIn(256).nOut(latentDim).build(), "encoder_hidden")
            .addLayer("log_std", new DenseLayer.Builder().nIn(256).nOut(latentDim).build(), "encoder_hidden")
            .addVertex("latent", new MergeVertex(), "mean", "log_std")
            .addLayer("decoder_hidden", new DenseLayer.Builder().nIn(latentDim).nOut(256).activation(Activation.RELU).build(), "latent")
            .addLayer("output", new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.SIGMOID).nIn(256).nOut(imageSize).build(), "decoder_hidden")
            .setOutputs("output")
            .setInputTypes(InputType.feedForward(imageSize))
            .build();

        return new ComputationGraph(conf);
    }


    public static void trainModel(ComputationGraph vae, MnistDataSetIterator data) throws Exception {
        
    	int count =0;
    	
    	while (data.hasNext()) {
            DataSet batch = data.next();
            INDArray features = batch.getFeatures();

            // Fetching outputs from specific layers
            Map<String, INDArray> layerOutputs = vae.feedForward(features, false);
            INDArray mean = layerOutputs.get("mean");
            INDArray logStd = layerOutputs.get("log_std");

            INDArray latent = reparameterize(mean, logStd);

            // Create a DataSet where features are both the inputs and the labels, typical for autoencoders
            DataSet autoencoderInput = new DataSet(features, features);
            vae.fit(autoencoderInput);
            
            if( count % 10 == 0 )
            {
            	INDArray  sampledImages = sampleFromLatentSpace(vae, 10,784);
                visualize(sampledImages);
                
            }
          
            count++;
            System.out.println(count);

        }
    }

    public static INDArray reparameterize(INDArray mean, INDArray logStd) {
        INDArray epsilon = Nd4j.randn(mean.shape());
        return mean.add(epsilon.mul(Transforms.exp(logStd, true)));
    }

    public static void visualize(INDArray images) {
        JFrame frame = new JFrame("Sampled MNIST Images from VAE");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setLayout(new GridLayout(1, images.rows()));

        for (int i = 0; i < images.rows(); i++) {
            frame.add(createImagePanel(images.getRow(i)));
        }

        frame.pack();
        frame.setVisible(true);
    }


    private static JPanel createImagePanel(INDArray imageRow) {
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
        panel.add(new JLabel(""), BorderLayout.NORTH);
        panel.add(new JLabel(new ImageIcon(img)), BorderLayout.CENTER);
        return panel;
    }
    
}
