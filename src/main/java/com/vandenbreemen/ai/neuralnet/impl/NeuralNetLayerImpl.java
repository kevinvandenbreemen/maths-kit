package com.vandenbreemen.ai.neuralnet.impl;

import com.vandenbreemen.ai.neuralnet.api.CostFunction;
import com.vandenbreemen.ai.neuralnet.api.NeuralNetLayer;
import com.vandenbreemen.ai.neuralnet.api.TrainingExample;
import com.vandenbreemen.linalg.api.LinalgProvider;
import com.vandenbreemen.linalg.api.Matrix;
import com.vandenbreemen.linalg.api.Vector;
import com.vandenbreemen.linalg.api.VectorFunction;
import com.vandenbreemen.linalg.impl.SigmoidDerivative;
import com.vandenbreemen.linalg.impl.SigmoidFunction;

public class NeuralNetLayerImpl implements NeuralNetLayer {

    private Vector biasVector;
    private Matrix weightMatrix;
    private LinalgProvider linalgProvider;

    private VectorFunction activationFunction;

    private VectorFunction activationFunctionDerivative;

    private CostFunction costFunction;

    public NeuralNetLayerImpl(LinalgProvider linalgProvider, int numInputs, int numOutputs){
        this.linalgProvider = linalgProvider;
        this.biasVector = linalgProvider.getVector(new double[numOutputs]);
        this.biasVector = linalgProvider.getOperations().randomEntries(biasVector);

        this.weightMatrix = this.linalgProvider.getMatrix(numOutputs, numInputs);
        this.weightMatrix = this.linalgProvider.getOperations().randomEntries(weightMatrix);

        this.activationFunction = new SigmoidFunction();
        this.activationFunctionDerivative = new SigmoidDerivative((SigmoidFunction)this.activationFunction);
        this.costFunction = new QuadraticCostFunction(linalgProvider);
    }

    @Override
    public Vector getActivation(Vector input) {
        Vector weightedInputs = getWeightedInput(input);
        return linalgProvider.getOperations().function(weightedInputs, activationFunction);
    }

    @Override
    public Vector getWeightedInput(Vector input) {
        Vector weightedInputs = linalgProvider.getOperations().matrixVectorProduct(weightMatrix, input);
        weightedInputs = linalgProvider.getOperations().add(weightedInputs, biasVector);
        return weightedInputs;
    }

    @Override
    public Vector getDerivativeOfActivation(Vector input) {
        Vector weightedInputs = getWeightedInput(input);
        return linalgProvider.getOperations().function(weightedInputs, activationFunctionDerivative);
    }

    @Override
    public Vector getOutputError(TrainingExample example){
        if(example.getActualOutput() == null){
            throw new RuntimeException("Missing:  Actual Output to compare against");
        }

        Vector errorGradient = costFunction.gradient(example);
        Vector weightedInput = getWeightedInput(example.getInput());
        Vector activationDerivative = linalgProvider.getOperations().function(weightedInput, activationFunctionDerivative);
        return linalgProvider.getOperations().hadamard(errorGradient, activationDerivative);
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
