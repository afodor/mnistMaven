package mnist;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.text.NumberFormat;

public class MNISTClassifier {

    public static void main(String[] args) throws Exception {
        int batchSize = 64;
        int rngSeed = 123;
        int numEpochs = 10;

        // Load the MNIST data
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, rngSeed);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, rngSeed);

        // Build the neural network configuration
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(rngSeed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam())
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(28 * 28)
                        .nOut(128)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nIn(128)
                        .nOut(2)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(2)
                        .nOut(10)
                        .activation(Activation.SOFTMAX)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .build();

        // Initialize the model
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10));

        // Train the model
        for (int epoch = 0; epoch < numEpochs; epoch++) {
        	
            // Evaluate the model
            Evaluation eval = new Evaluation(10);
            while (mnistTest.hasNext()) {
                DataSet ds = mnistTest.next();
                INDArray output = model.output(ds.getFeatures(), false);
                eval.eval(ds.getLabels(), output);
            }
            mnistTest.reset();
            System.out.println(eval.stats());
        	
        	System.out.println("Starting epoch " + epoch);
        	 writeLastHiddenLayerValuesToFile(model, mnistTest, epoch, eval.accuracy());
            model.fit(mnistTrain);
            mnistTrain.reset();
            
        }

    }

    public static void writeLastHiddenLayerValuesToFile(MultiLayerNetwork model, DataSetIterator dataSetIterator, int epoch, double accuracy)
    	throws Exception {
    	 NumberFormat nf = NumberFormat.getInstance();
    	 nf.setMaximumFractionDigits(3);
        System.out.println("Writing file for " + epoch);
    	BufferedWriter writer = new BufferedWriter(new FileWriter(new File("c:\\temp\\epoch_" + epoch + "_" + nf.format(accuracy) + ".txt")));
        try {
            dataSetIterator.reset();
            writer.write("Neuron1\tNeuron2\tLabel\n");
            
            int count =0;
            
            while (dataSetIterator.hasNext()) {
            	
            	if( count % 100 == 0 )
            	{
            		//System.out.println(count);
            		DataSet ds = dataSetIterator.next();
                    INDArray features = ds.getFeatures();
                    INDArray labels = ds.getLabels();
                    INDArray lastHiddenLayerOutput = model.feedForward(features).get(2); // Get the output of the last hidden layer

                    for (int i = 0; i < features.size(0); i++) {
                    	//if( i % 100 == 0 )
                    	{
                    		//System.out.println(i + " " + features.size(0));
                    		INDArray row = lastHiddenLayerOutput.getRow(i);
                            int label = labels.getRow(i).argMax().getInt(0);
                            writer.write(row.getDouble(0) + "\t" + row.getDouble(1) + "\t" + label + "\n");
                    	}
                    }
            	}
            	count++;
            	
            	
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (writer != null) {
                try {
                    writer.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
        
        System.out.println("Finished writing file");
    }
}
 
