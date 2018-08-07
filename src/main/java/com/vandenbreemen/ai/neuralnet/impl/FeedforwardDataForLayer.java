package com.vandenbreemen.ai.neuralnet.impl;


import com.vandenbreemen.linalg.api.Vector;

class FeedforwardDataForLayer {

    final Vector weightedInput;

    final Vector activations;

    /**
     * Error for the layer
     */
    Vector error;

    public FeedforwardDataForLayer(Vector weightedInput, Vector activations) {
        this.weightedInput = weightedInput;
        this.activations = activations;
    }
}
