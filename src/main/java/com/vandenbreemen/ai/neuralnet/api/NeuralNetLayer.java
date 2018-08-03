package com.vandenbreemen.ai.neuralnet.api;

import com.vandenbreemen.linalg.api.Vector;

public interface NeuralNetLayer {

    Vector getActivation(Vector input);

    Vector getWeightedInput(Vector input);

    Vector getDerivativeOfActivation(Vector input);

    Vector getOutputError(TrainingExample example);

    int getNumInputs();

    int getNumOutputs();
}
