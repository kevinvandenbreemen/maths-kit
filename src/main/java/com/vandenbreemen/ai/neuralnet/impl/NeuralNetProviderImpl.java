package com.vandenbreemen.ai.neuralnet.impl;

import com.vandenbreemen.ai.neuralnet.api.NeuralNet;
import com.vandenbreemen.ai.neuralnet.api.NeuralNetLayer;
import com.vandenbreemen.ai.neuralnet.api.NeuralNetProvider;
import com.vandenbreemen.linalg.api.LinalgProvider;

public class NeuralNetProviderImpl implements NeuralNetProvider {

    private LinalgProvider linalgProvider;

    public NeuralNetProviderImpl(LinalgProvider linalgProvider) {
        this.linalgProvider = linalgProvider;
    }

    @Override
    public NeuralNet getNeuralNet(int numInputs) {
        return new NeuralNetImpl(linalgProvider, numInputs);
    }

    @Override
    public NeuralNetLayer createLayer(NeuralNet net, int numOutputs) {
        return new NeuralNetLayerImpl(linalgProvider,
                net.getNumOutputs() == 0 ? net.getNumInputs(): net.getNumOutputs(), numOutputs);
    }
}
