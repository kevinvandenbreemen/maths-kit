package com.vandenbreemen.ai.neuralnet.impl;

import com.vandenbreemen.ai.neuralnet.api.NeuralNet;
import com.vandenbreemen.ai.neuralnet.api.NeuralNetLayer;

import java.util.List;

public class NeuralNetImpl implements NeuralNet {

    private int numInputs;

    private NeuralNetLayer[] layers;

    public NeuralNetImpl(int numInputs) {
        this.numInputs = numInputs;
        this.layers = new NeuralNetLayer[0];
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
}
