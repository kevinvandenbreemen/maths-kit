package com.vandenbreemen.ai.neuralnet.impl;

import com.vandenbreemen.ai.neuralnet.api.CostFunction;
import com.vandenbreemen.ai.neuralnet.api.NeuralNet;
import com.vandenbreemen.ai.neuralnet.api.NeuralNetLayer;
import com.vandenbreemen.ai.neuralnet.api.TrainingExample;
import com.vandenbreemen.linalg.api.LinalgProvider;
import com.vandenbreemen.linalg.api.Vector;
import com.vandenbreemen.linalg.api.VectorFunction;
import com.vandenbreemen.linalg.impl.SigmoidDerivative;
import com.vandenbreemen.linalg.impl.SigmoidFunction;

public class NeuralNetImpl implements NeuralNet {

    private int numInputs;

    private LinalgProvider linalgProvider;

    private NeuralNetLayer[] layers;

    private CostFunction costFunction;

    private VectorFunction activationFunctionDerivative;

    public NeuralNetImpl(LinalgProvider linalgProvider, int numInputs) {
        this.linalgProvider = linalgProvider;
        this.numInputs = numInputs;
        this.layers = new NeuralNetLayer[0];
        this.costFunction = new QuadraticCostFunction(this.linalgProvider);
        this.activationFunctionDerivative = new SigmoidDerivative(new SigmoidFunction());
    }

    @Override
    public int getNumInputs() {
        return numInputs;
    }

    @Override
    public void addLayer(NeuralNetLayer layer) {
        NeuralNetLayer[] newLayers = new NeuralNetLayer[this.layers.length+1];
        System.arraycopy(layers, 0, newLayers, 0, this.layers.length);
        newLayers[layers.length] = layer;
        this.layers = newLayers;
    }

    @Override
    public int getNumOutputs() {
        return this.layers.length == 0 ? 0:
                this.layers[this.layers.length-1].getNumOutputs();
    }

    @Override
    public Vector getOutout(Vector input) {
        Vector activations = linalgProvider.copyVector(input);
        for(NeuralNetLayer layer : this.layers){
            activations = layer.getActivation(activations);
        }

        return activations;
    }

    /**
     * Compute the z term for the output layer of the net
     * @param example
     * @return
     */
    private Vector getWeightedInput(TrainingExample example){
        Vector activations = linalgProvider.copyVector(example.getInput());

        for(int i=0; i<layers.length; i++){
            if(i < layers.length-1) {
                activations = layers[i].getActivation(activations);
            }
        }

        //  Now we're at final layer, so output the weighted input for that layer!
        return layers[layers.length-1].getWeightedInput(activations);
    }

    @Override
    public void train(TrainingExample... examples) {
        //  Basic Gradient Descent
        for(TrainingExample example:examples){
            example.setActualOutput(getOutout(example.getInput()));
            Vector outputError = getOutputError(example);   //  delta_L
            System.out.println("error="+outputError);
        }
    }

    private Vector getOutputError(TrainingExample example) {
        Vector costGradient = this.costFunction.gradient(example);
        Vector activationDerivative = getWeightedInput(example);
        activationDerivative = linalgProvider.getOperations().function(activationDerivative, this.activationFunctionDerivative);
        return linalgProvider.getOperations().hadamard(costGradient, activationDerivative);
    }
}
