package simpleSampleData;

import java.io.*;
import java.text.DecimalFormat;
import java.util.Random;

public class SimpleNeuralNetwork {

    private static final int IMAGE_SIZE = 28 * 28;
    private static final int NUM_CLASSES = 2;
    private static final int NUM_EPOCHS = 10;
    private static final int BATCH_SIZE = 10;
    private static final double LEARNING_RATE = 0.01;

    private static final int HIDDEN_LAYER_SIZE = 64;

    private double[][] weightsInputToHidden;
    private double[][] weightsHiddenToOutput;

    private Random random;

    public SimpleNeuralNetwork() {
        random = new Random();
        weightsInputToHidden = new double[IMAGE_SIZE][HIDDEN_LAYER_SIZE];
        weightsHiddenToOutput = new double[HIDDEN_LAYER_SIZE][NUM_CLASSES];

        // Initialize weights
        for (int i = 0; i < IMAGE_SIZE; i++) {
            for (int j = 0; j < HIDDEN_LAYER_SIZE; j++) {
                weightsInputToHidden[i][j] = random.nextGaussian() * 0.01;
            }
        }

        for (int i = 0; i < HIDDEN_LAYER_SIZE; i++) {
            for (int j = 0; j < NUM_CLASSES; j++) {
                weightsHiddenToOutput[i][j] = random.nextGaussian() * 0.01;
            }
        }
    }

    public static void main(String[] args) throws Exception {
        SimpleNeuralNetwork nn = new SimpleNeuralNetwork();
        int[][] images = readMNISTImages("C:\\temp\\synthetic_images.idx3-ubyte");
        int[] labels = readMNISTLabels("C:\\temp\\synthetic_labels.idx1-ubyte");

        nn.train(images, labels);
        double accuracy = nn.evaluate(images, labels);
        System.out.println("Accuracy: " + accuracy * 100 + "%");

        nn.writeOutputValues("c:\\temp\\output_values_FINAL.txt", images, labels);
    }

    private static String getPrettyCount(int count)
    {
    	String val = "" + count;
    	
    	while( val.length() <3)
    		val = "0" + val;
    	
    	return val;
    }
    
    private void train(int[][] images, int[] labels) throws Exception {
    	int count =0;
    	
    	 DecimalFormat df = new DecimalFormat("#.00");
    	
        for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
            for (int i = 0; i < images.length; i++) 
            {
            	if( count < 100)
            	{
            		this.writeOutputValues("c:\\temp\\output_values_" + getPrettyCount(count) + 
                    		"_" + df.format(this.evaluate(images, labels)) + 	".txt", images,labels);
                    	
            	}
            	
            	count++;
            	
                int[] image = images[i];
                int label = labels[i];

                // Forward pass
                double[] hiddenLayer = new double[HIDDEN_LAYER_SIZE];
                for (int j = 0; j < HIDDEN_LAYER_SIZE; j++) {
                    hiddenLayer[j] = 0.0;
                    for (int k = 0; k < IMAGE_SIZE; k++) {
                        hiddenLayer[j] += image[k] * weightsInputToHidden[k][j];
                    }
                    hiddenLayer[j] = sigmoid(hiddenLayer[j]);
                }

                double[] outputs = new double[NUM_CLASSES];
                for (int j = 0; j < NUM_CLASSES; j++) {
                    outputs[j] = 0.0;
                    for (int k = 0; k < HIDDEN_LAYER_SIZE; k++) {
                        outputs[j] += hiddenLayer[k] * weightsHiddenToOutput[k][j];
                    }
                    outputs[j] = sigmoid(outputs[j]);
                }

                // Compute errors
                double[] target = new double[NUM_CLASSES];
                target[label] = 1.0;
                double[] errors = new double[NUM_CLASSES];
                for (int j = 0; j < NUM_CLASSES; j++) {
                    errors[j] = outputs[j] - target[j];
                }

                // Backward pass
                double[] hiddenLayerGradient = new double[HIDDEN_LAYER_SIZE];
                for (int j = 0; j < HIDDEN_LAYER_SIZE; j++) {
                    hiddenLayerGradient[j] = 0.0;
                    for (int k = 0; k < NUM_CLASSES; k++) {
                        hiddenLayerGradient[j] += errors[k] * sigmoidDerivative(outputs[k]) * weightsHiddenToOutput[j][k];
                        weightsHiddenToOutput[j][k] -= LEARNING_RATE * errors[k] * sigmoidDerivative(outputs[k]) * hiddenLayer[j];
                    }
                }

                for (int j = 0; j < HIDDEN_LAYER_SIZE; j++) {
                    for (int k = 0; k < IMAGE_SIZE; k++) {
                        weightsInputToHidden[k][j] -= LEARNING_RATE * hiddenLayerGradient[j] * sigmoidDerivative(hiddenLayer[j]) * image[k];
                    }
                }
            }
            System.out.println("Epoch " + (epoch + 1) + " complete.");
            this.writeOutputValues("c:\\temp\\output_values_" + "EPOCH_" + getPrettyCount(epoch+1)+  
            		"_" + df.format(this.evaluate(images, labels)) + 	".txt", images,labels);
        }
    }

    private double evaluate(int[][] images, int[] labels) {
        int correct = 0;
        for (int i = 0; i < images.length; i++) {
            if (predict(images[i]) == labels[i]) {
                correct++;
            }
        }
        return (double) correct / images.length;
    }

    private int predict(int[] image) {
        double[] outputs = forwardPass(image);
        return (outputs[0] >= outputs[1]) ? 0 : 1;
    }

    private double[] forwardPass(int[] image) {
        double[] hiddenLayer = new double[HIDDEN_LAYER_SIZE];
        for (int j = 0; j < HIDDEN_LAYER_SIZE; j++) {
            hiddenLayer[j] = 0.0;
            for (int k = 0; k < IMAGE_SIZE; k++) {
                hiddenLayer[j] += image[k] * weightsInputToHidden[k][j];
            }
            hiddenLayer[j] = sigmoid(hiddenLayer[j]);
        }

        double[] outputs = new double[NUM_CLASSES];
        for (int j = 0; j < NUM_CLASSES; j++) {
            outputs[j] = 0.0;
            for (int k = 0; k < HIDDEN_LAYER_SIZE; k++) {
                outputs[j] += hiddenLayer[k] * weightsHiddenToOutput[k][j];
            }
            outputs[j] = sigmoid(outputs[j]);
        }

        return outputs;
    }

    private void writeOutputValues(String fileName, int[][] images, int[] labels) throws Exception
    {
        	BufferedWriter writer = new BufferedWriter(new FileWriter(fileName));
        	writer.write("V1\tV2\tlabel\n");
        	int count =0;
            for (int[] image : images) {
                double[] outputs = forwardPass(image);
                writer.write(outputs[0] + "\t" + outputs[1] + "\t" + labels[count] + "\n");
                count++;
            }
            writer.flush(); writer.close();
    }

    private static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    private static double sigmoidDerivative(double x) {
        return x * (1.0 - x);
    }

    private static int[][] readMNISTImages(String fileName) throws Exception {
        DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(fileName)));
            int magicNumber = dis.readInt();
            int numberOfImages = dis.readInt();
            int numberOfRows = dis.readInt();
            int numberOfColumns = dis.readInt();

            int[][] images = new int[numberOfImages][numberOfRows * numberOfColumns];
            for (int i = 0; i < numberOfImages; i++) {
                for (int j = 0; j < numberOfRows * numberOfColumns; j++) {
                    images[i][j] = dis.readUnsignedByte();
                }
            }
            return images;
    }

    private static int[] readMNISTLabels(String fileName) throws IOException {
    	DataInputStream dis = new DataInputStream(new BufferedInputStream(new FileInputStream(fileName)));
            int magicNumber = dis.readInt();
            int numberOfLabels = dis.readInt();

            int[] labels = new int[numberOfLabels];
            for (int i = 0; i < numberOfLabels; i++) {
                labels[i] = dis.readUnsignedByte();
            }
            return labels;
        }
}
