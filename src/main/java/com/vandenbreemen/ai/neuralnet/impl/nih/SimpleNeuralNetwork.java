package com.vandenbreemen.ai.neuralnet.impl.nih;

import com.vandenbreemen.linalg.api.LinalgProvider;
import com.vandenbreemen.linalg.api.Matrix;
import com.vandenbreemen.linalg.api.Vector;
import com.vandenbreemen.linalg.api.VectorFunction;
import com.vandenbreemen.linalg.impl.LinalgProviderImpl;

import java.util.ArrayList;
import java.util.List;

public class SimpleNeuralNetwork {

    /**
     * Sigmoid function by default
     */
    private VectorFunction activationFunction = (entry)-> 1.0 / (1.0 + Math.exp(-1.0 * entry));

    /**
     * Weight matrix for each layer
     */
    private List<Matrix> layerWeightMatrices;

    /**
     * Bias vector column for each layer
     */
    private List<Vector> layerBiases;

    private LinalgProvider linalgProvider;

    private boolean debug = false;

    public SimpleNeuralNetwork(LinalgProvider linalgProvider) {
        this.linalgProvider = linalgProvider;
        this.layerWeightMatrices = new ArrayList<>();
        this.layerBiases = new ArrayList<>();
    }

    public SimpleNeuralNetwork setDebug(boolean debug) {
        this.debug = debug;
        return this;
    }

    /**
     * Add a layer with the given number of inputs/outputs
     * @param inputs
     * @param outputs
     */
    public void addLayer(int inputs, int outputs){
        Vector biasColumn = linalgProvider.vectorOf(0.0, outputs);
        biasColumn = linalgProvider.getOperations().randomEntries(biasColumn);

        Matrix weightMatrix = linalgProvider.getMatrix(outputs, inputs);
        weightMatrix = linalgProvider.getOperations().randomEntries(weightMatrix);

        this.layerBiases.add(biasColumn);
        this.layerWeightMatrices.add(weightMatrix);
    }

    public Matrix feedForwardResults(Matrix inputs){
        if(inputs.rows() != layerWeightMatrices.get(0).cols()){
            throw new RuntimeException("Missing input units");
        }

        return doFeedForward(inputs, 0);
    }

    private Matrix doFeedForward(Matrix inputs, int layerIndex) {

        if(debug){
            System.err.println("Feeding inputs:\n"+inputs+"\n to layer "+layerIndex +" (layer = \n"+layerWeightMatrices.get(layerIndex));
        }

        Matrix inputsPrime = linalgProvider.getOperations().transpose(inputs);
        linalgProvider.getOperations().prependColumn(inputsPrime, linalgProvider.vectorOf(1.0, inputsPrime.rows()));
        inputsPrime = linalgProvider.getOperations().transpose(inputsPrime);

        Matrix layerPrime = linalgProvider.getOperations().copy(layerWeightMatrices.get(layerIndex));
        linalgProvider.getOperations().prependColumn(layerPrime, layerBiases.get(layerIndex));

        Matrix zMatrix = linalgProvider.getOperations().matrixMatrixProduct(layerPrime, inputsPrime);
        Matrix activations = linalgProvider.getOperations().function(zMatrix, activationFunction);

        if(layerIndex >= layerWeightMatrices.size()-1){
            return activations;
        }
        return doFeedForward(activations, ++layerIndex);
    }

    public static void main(String[] args){

        LinalgProvider provider = new LinalgProviderImpl();

        Matrix xorExamples = provider.getMatrix( new double[][]{
                        new double[]{1,1,0,0},
                        new double[]{1,0,1,0},
                }
        );

        SimpleNeuralNetwork network = new SimpleNeuralNetwork(provider);
        network.addLayer(2, 3);
        network.addLayer(3, 1);

        System.out.println(network.feedForwardResults(xorExamples));
    }

}
