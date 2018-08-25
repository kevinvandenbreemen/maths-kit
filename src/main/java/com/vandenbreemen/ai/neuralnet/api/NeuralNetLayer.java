package com.vandenbreemen.ai.neuralnet.api;

import com.vandenbreemen.linalg.api.Matrix;
import com.vandenbreemen.linalg.api.Vector;

public interface NeuralNetLayer {

    Vector getActivation(Vector input);

    Vector getWeightedInput(Vector input);

    Matrix getWeightMatrixTranspose();

    int getNumInputs();

    int getNumOutputs();
}
