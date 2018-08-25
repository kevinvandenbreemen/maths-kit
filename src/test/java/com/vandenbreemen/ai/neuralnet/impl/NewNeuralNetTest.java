package com.vandenbreemen.ai.neuralnet.impl;

import com.vandenbreemen.ai.neuralnet.api.NeuralNet;
import com.vandenbreemen.linalg.api.LinalgProvider;
import com.vandenbreemen.linalg.api.Vector;
import com.vandenbreemen.linalg.impl.LinalgProviderImpl;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

public class NewNeuralNetTest {

    private LinalgProvider linalgProvider;

    @Before
    public void setup(){
        this.linalgProvider = new LinalgProviderImpl();
    }

    @Test
    public void shouldGetActivation(){
        //  Arrange
        NewNeuralNet net = new NewNeuralNet(linalgProvider);
        net.addLayer(2, 3);
        net.addLayer(3, 2);

        //  Act
        System.out.println(net.compute_hTheta(linalgProvider.getVector(new double[]{1.0, 0.0})));

        //  Assert
    }

    @Test
    public void shouldSetWeights(){

        //  Arrange
        NewNeuralNet net = createXORNetWithPredictableWeights();

        System.out.println(net);
        System.out.println(net.compute_hTheta(linalgProvider.getVector(new double[]{1.0, 0.0})));

    }

    @Test
    public void shouldDoTheFuckingBackProp(){
        NewNeuralNet net = createXORNetWithPredictableWeights();

    }

    private NewNeuralNet createXORNetWithPredictableWeights() {
        NewNeuralNet net = new NewNeuralNet(linalgProvider);
        net.addLayer(2, 3);
        net.addLayer(3, 2);

        //  Input layer to hidden layer
        net.setWeight(0, 0, 0, 0.9);
        net.setWeight(0, 0, 1, 0.2);
        net.setWeight(0, 0, 2, 0.3);

        net.setWeight(0, 1, 0, 0.7);
        net.setWeight(0, 1, 1, 0.6);
        net.setWeight(0, 1, 2, 0.3);

        net.setWeight(0, 2, 0, 0.2);
        net.setWeight(0, 2, 1, 0.3);
        net.setWeight(0, 2, 2, 0.4);

        //  Hidden layer to output
        net.setWeight(1, 0, 0, 0.1);
        net.setWeight(1, 0, 1, 0.2);

        net.setWeight(1, 1, 0, 0.11);
        net.setWeight(1, 1, 1, 0.21);

        net.setWeight(1, 2, 0, 0.13);
        net.setWeight(1, 2, 1, 0.23);

        net.setWeight(1, 3, 0, 0.14);
        net.setWeight(1, 3, 1, 0.24);
        return net;
    }

}