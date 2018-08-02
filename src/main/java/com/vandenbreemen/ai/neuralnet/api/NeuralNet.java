package com.vandenbreemen.ai.neuralnet.api;

import com.vandenbreemen.linalg.api.Vector;

public interface NeuralNet {

    int getNumInputs();

    void addLayer(NeuralNetLayer layer);

    int getNumOutputs();

    Vector getOutout(Vector input);
}
