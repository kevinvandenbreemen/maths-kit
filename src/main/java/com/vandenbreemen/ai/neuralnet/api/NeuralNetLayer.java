package com.vandenbreemen.ai.neuralnet.api;

import com.vandenbreemen.linalg.api.Vector;

public interface NeuralNetLayer {

    Vector getActivation(Vector input);

    int getNumInputs();

    int getNumOutputs();
}
