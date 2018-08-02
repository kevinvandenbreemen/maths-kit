package com.vandenbreemen.ai.neuralnet.api;

public interface NeuralNet {

    int getNumInputs();

    void addLayer(NeuralNetLayer layer);

    int getNumOutputs();
}
