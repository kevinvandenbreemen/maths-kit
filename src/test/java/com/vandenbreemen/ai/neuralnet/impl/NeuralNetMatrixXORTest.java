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

        theta1 = linalgProvider.getMatrix(4, 3);    //  2 inputs (columns), 3 outputs (hidden layers)
        theta2 = linalgProvider.getMatrix(1, 3);    //  3 inputs (columns), 1 outputs (output)

        trainingInputs = linalgProvider.getMatrix(new double[][]{
                new double[]{1, 1, 0, 0},
                new double[]{1, 0, 1, 0}
        });

        trainingOutputs = linalgProvider.getVector(new double[]{0,1,1,0});

        linalgProvider.getOperations().randomEntries(theta1);
        linalgProvider.getOperations().randomEntries(theta2);
    }

    @Test
    public void shouldRandomInit(){
        linalgProvider.getOperations().randomEntries(theta1);
        linalgProvider.getOperations().randomEntries(theta2);


    }

    @Test
    public void shouldComputeActivationsOfHiddenLayer(){

        Matrix samplesTranspose = linalgProvider.getOperations().transpose(trainingInputs);
        linalgProvider.getOperations().prependColumn(samplesTranspose, linalgProvider.vectorOf(1.0, samplesTranspose.rows()));
        System.out.println(samplesTranspose);
        trainingInputs = linalgProvider.getOperations().transpose(samplesTranspose);
        System.out.println(trainingInputs);


        Matrix z_2Matrix = linalgProvider.getOperations().matrixMatrixProduct(theta1, trainingInputs);
        System.out.println("Z Layer 2:\n"+z_2Matrix);

        Matrix activationsLayer2 = linalgProvider.getOperations().function(z_2Matrix, (entry)->
            1.0 / (1.0 + Math.exp(-1.0 * entry))
        );
        System.out.println("ACTIVATIONS:\n"+activationsLayer2);
    }

    @Test
    public void shouldComputeActivationsOfOutputLayer(){

    }


}
