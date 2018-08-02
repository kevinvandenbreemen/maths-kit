package com.vandenbreemen.ai.neuralnet.impl;

import com.vandenbreemen.ai.neuralnet.api.NeuralNet;
import com.vandenbreemen.ai.neuralnet.api.NeuralNetLayer;
import com.vandenbreemen.ai.neuralnet.api.NeuralNetProvider;
import com.vandenbreemen.linalg.impl.LinalgProviderImpl;
import org.junit.Before;
import org.junit.Test;

import static junit.framework.TestCase.assertEquals;

public class NeuralNetImplTest {

    private NeuralNetProvider neuralNetProvider;

    @Before
    public void setup(){
        this.neuralNetProvider = new NeuralNetProviderImpl(new LinalgProviderImpl());
    }

    @Test
    public void shouldConstruct(){
        NeuralNet net = neuralNetProvider.getNeuralNet(2);
    }

    @Test
    public void shouldAllowForNewLayers(){
        //  Arrange
        NeuralNet net = neuralNetProvider.getNeuralNet(2);

        //  Act
        NeuralNetLayer layer = neuralNetProvider.createLayer(net, 3);

        //  Assert
        assertEquals("Layer Inputs", layer.getNumInputs(), 2);
        assertEquals("Layer Outputs", 3, layer.getNumOutputs());
    }

    @Test
    public void shouldAddLayersAfterFirstLayer(){
        //  Arrange
        NeuralNet net = neuralNetProvider.getNeuralNet(2);
        net.addLayer(neuralNetProvider.createLayer(net, 3));

        //  Act
        NeuralNetLayer layer = neuralNetProvider.createLayer(net, 5);

        //  Assert
        assertEquals("Layer Inputs", layer.getNumInputs(), 3);
        assertEquals("Layer Outputs", 5, layer.getNumOutputs());
    }

}