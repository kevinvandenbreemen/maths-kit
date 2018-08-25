package com.vandenbreemen.ai.neuralnet.impl;

import com.vandenbreemen.linalg.api.LinalgProvider;
import com.vandenbreemen.linalg.api.Vector;
import com.vandenbreemen.linalg.impl.LinalgProviderImpl;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

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



        List<Vector> trainingInputs = new ArrayList<>();
        trainingInputs.add(linalgProvider.getVector(new double[]{1,1}));
        trainingInputs.add(linalgProvider.getVector(new double[]{1,0}));
        trainingInputs.add(linalgProvider.getVector(new double[]{0,1}));
        trainingInputs.add(linalgProvider.getVector(new double[]{0,0}));

        List<Vector> expectedOutputs = new ArrayList<>();
        expectedOutputs.add(linalgProvider.getVector(new double[]{0,1}));
        expectedOutputs.add(linalgProvider.getVector(new double[]{1,0}));
        expectedOutputs.add(linalgProvider.getVector(new double[]{1,0}));
        expectedOutputs.add(linalgProvider.getVector(new double[]{0,1}));

        List<Vector> beforeTraining = new ArrayList<>();
        for (Vector input : trainingInputs){
            beforeTraining.add(net.compute_hTheta(input));
        }

        net.train(trainingInputs, expectedOutputs);

        System.out.println("RESULTS AFTER TRAINING");
        for (Vector input : trainingInputs){
            System.out.println("input:  "+input);
            System.out.println("output:  "+net.compute_hTheta(input));
        }

        System.out.println("VS:\n"+beforeTraining);

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