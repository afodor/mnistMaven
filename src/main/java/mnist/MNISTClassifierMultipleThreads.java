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

import sun.jvm.hotspot.debugger.cdbg.basic.BasicEnumType;

import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.deeplearning4j.parallelism.ParallelWrapper;

public class MNISTClassifierMultipleThreads {

    public static void main(String[] args) throws Exception {
        int batchSize = 64;
        int rngSeed = 123;
        int numEpochs = 5;
        int numWorkers = Runtime.getRuntime().availableProcessors()-1; // Set number of workers to the number of available processors

        // Set environment variables for threading
        System.setProperty("OMP_NUM_THREADS", String.valueOf(numWorkers));
        System.setProperty("MKL_NUM_THREADS", String.valueOf(numWorkers));
        System.out.println("Starting with " + numWorkers);
        
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

        // Create ParallelWrapper
        ParallelWrapper wrapper = new ParallelWrapper.Builder(model)
                .prefetchBuffer(24)
                .workers(numWorkers) // Number of threads for training
                .averagingFrequency(1)
                .reportScoreAfterAveraging(false)
                .build();

        // Train the model
        for (int i = 0; i < numEpochs; i++) {
            wrapper.fit(mnistTrain);
            mnistTrain.reset();
        }

        // Evaluate the model
        Evaluation eval = new Evaluation(10);
        while (mnistTest.hasNext()) {
            DataSet ds = mnistTest.next();
            INDArray output = model.output(ds.getFeatures(), false);
            eval.eval(ds.getLabels(), output);
        }
        mnistTest.reset();

        System.out.println(eval.stats());
    }
}
