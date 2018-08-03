package com.vandenbreemen.ai.neuralnet.impl;

import com.vandenbreemen.ai.neuralnet.api.NeuralNet;
import com.vandenbreemen.ai.neuralnet.api.NeuralNetLayer;
import com.vandenbreemen.ai.neuralnet.api.NeuralNetProvider;
import com.vandenbreemen.ai.neuralnet.api.TrainingExample;
import com.vandenbreemen.linalg.api.LinalgProvider;
import com.vandenbreemen.linalg.api.Vector;
import com.vandenbreemen.linalg.impl.LinalgProviderImpl;
import org.junit.Before;
import org.junit.Test;

import static junit.framework.TestCase.assertEquals;

public class NeuralNetImplTest {

    private NeuralNetProvider neuralNetProvider;

    private LinalgProvider linalgProvider;

    @Before
    public void setup(){
        this.linalgProvider = new LinalgProviderImpl();
        this.neuralNetProvider = new NeuralNetProviderImpl(linalgProvider);
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

    @Test
    public void shouldFeedForward(){
        //  Arrange
        NeuralNet net = neuralNetProvider.getNeuralNet(2);
        net.addLayer(neuralNetProvider.createLayer(net, 3));
        net.addLayer(neuralNetProvider.createLayer(net, 2));

        //  Act
        Vector output = net.getOutout(
            linalgProvider.getVector(new double[]{0.0, 1.0})
        );

        //  Do something
        System.out.println(output);
        assertEquals("Entries", 2, output.length());
    }

    @Test
    public void shouldBackPropagateWithoutCrash(){
        //  Arrange
        NeuralNet net = neuralNetProvider.getNeuralNet(2);
        net.addLayer(neuralNetProvider.createLayer(net, 3));
        net.addLayer(neuralNetProvider.createLayer(net, 2));

        //  Act
        TrainingExample example = new TrainingExample(
                linalgProvider.getVector(new double[]{0., 1.}),
                linalgProvider.getVector(new double[]{1., 0.})
        );
        net.train(example);
    }

}