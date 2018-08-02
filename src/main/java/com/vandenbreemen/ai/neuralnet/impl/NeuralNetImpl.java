package com.vandenbreemen.ai.neuralnet.impl;

import com.vandenbreemen.ai.neuralnet.api.NeuralNet;
import com.vandenbreemen.ai.neuralnet.api.NeuralNetLayer;
import com.vandenbreemen.linalg.api.LinalgProvider;
import com.vandenbreemen.linalg.api.Vector;
import sun.plugin.javascript.navig4.Layer;

import java.util.List;

public class NeuralNetImpl implements NeuralNet {

    private int numInputs;

    private LinalgProvider linalgProvider;

    private NeuralNetLayer[] layers;

    public NeuralNetImpl(LinalgProvider linalgProvider, int numInputs) {
        this.linalgProvider = linalgProvider;
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

    @Override
    public Vector getOutout(Vector input) {
        Vector activations = linalgProvider.copyVector(input);
        for(NeuralNetLayer layer : this.layers){
            activations = layer.getActivation(activations);
        }

        return activations;
    }
}
