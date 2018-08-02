package com.vandenbreemen.ai.neuralnet.impl;

import com.vandenbreemen.ai.neuralnet.api.NeuralNetLayer;
import com.vandenbreemen.linalg.api.LinalgProvider;
import com.vandenbreemen.linalg.api.Matrix;
import com.vandenbreemen.linalg.api.Vector;
import com.vandenbreemen.linalg.api.VectorFunction;
import com.vandenbreemen.linalg.impl.SigmoidFunction;

public class NeuralNetLayerImpl implements NeuralNetLayer {

    private Vector biasVector;
    private Matrix weightMatrix;
    private LinalgProvider linalgProvider;

    private VectorFunction activationFunction;

    public NeuralNetLayerImpl(LinalgProvider linalgProvider, int numInputs, int numOutputs){
        this.linalgProvider = linalgProvider;
        this.biasVector = linalgProvider.getVector(new double[numOutputs]);
        this.biasVector = linalgProvider.getOperations().randomEntries(biasVector);

        this.weightMatrix = this.linalgProvider.getMatrix(numOutputs, numInputs);
        this.weightMatrix = this.linalgProvider.getOperations().randomEntries(weightMatrix);

        this.activationFunction = new SigmoidFunction();
    }

    @Override
    public Vector getActivation(Vector input) {
        Vector weightedInputs = linalgProvider.getOperations().matrixVectorProduct(weightMatrix, input);
        weightedInputs = linalgProvider.getOperations().add(weightedInputs, biasVector);
        return linalgProvider.getOperations().function(weightedInputs, activationFunction);
    }

    @Override
    public int getNumInputs() {
        return weightMatrix.cols();
    }

    @Override
    public int getNumOutputs() {
        return weightMatrix.rows();
    }
}
