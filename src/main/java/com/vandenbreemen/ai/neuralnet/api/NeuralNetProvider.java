package com.vandenbreemen.ai.neuralnet.api;

public interface NeuralNetProvider {

    /**
     * Create a new neural net with the given number of inputs
     * @param numInputs
     * @return
     */
    NeuralNet getNeuralNet(int numInputs);

    /**
     * Creates a new neural net layer.  Implementation should not add the layer to the neural net just yet
     * @param net
     * @param numOutputs
     * @return
     */
    NeuralNetLayer createLayer(NeuralNet net, int numOutputs);
}
