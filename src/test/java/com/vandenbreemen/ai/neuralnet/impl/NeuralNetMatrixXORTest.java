package com.vandenbreemen.ai.neuralnet.impl;

import com.vandenbreemen.linalg.api.LinalgProvider;
import com.vandenbreemen.linalg.api.Matrix;
import com.vandenbreemen.linalg.api.Vector;
import com.vandenbreemen.linalg.impl.LinalgProviderImpl;
import org.junit.Before;
import org.junit.Test;

/**
 * This will be a test to demonstrate back propagation implemented using matrices
 */
public class NeuralNetMatrixXORTest {

    private LinalgProvider linalgProvider;

    /**
     * The input layer -> hidden layer
     */
    private Matrix theta1;

    /**
     * The hidden layer -> output layer
     */
    private Matrix theta2;

    private Matrix trainingInputs;

    private Vector trainingOutputs;

    @Before
    public void setup(){

        linalgProvider = new LinalgProviderImpl();

        theta1 = linalgProvider.getMatrix(3, 2);    //  2 inputs (columns), 3 outputs (hidden layers)
        theta2 = linalgProvider.getMatrix(1, 3);    //  3 inputs (columns), 1 outputs (output)

        trainingInputs = linalgProvider.getMatrix(new double[][]{
                new double[]{1,1},
                new double[]{1,0},
                new double[]{0,1},
                new double[]{0,0}
        });

        trainingOutputs = linalgProvider.getVector(new double[]{0,1,1,0});
    }

    @Test
    public void shouldRandomInit(){
        linalgProvider.getOperations().randomEntries(theta1);
        linalgProvider.getOperations().randomEntries(theta2);


    }

    @Test
    public void shouldComputeActivationsOfHiddenLayer(){
        Vector z_2 = linalgProvider.getOperations().
    }


}
