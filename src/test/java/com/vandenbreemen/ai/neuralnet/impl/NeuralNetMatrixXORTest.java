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

    private Matrix trainingOutputs;

    @Before
    public void setup(){

        linalgProvider = new LinalgProviderImpl();

        theta1 = linalgProvider.getMatrix(3, 3);    //  2 inputs + 1 bias, outputs to 3 hidden units + 1 bias
        theta2 = linalgProvider.getMatrix(1, 4);    //  3 inputs + 1 bias(columns), 1 outputs (output)

        trainingInputs = linalgProvider.getMatrix(new double[][]{
                new double[]{1, 1, 0, 0},
                new double[]{1, 0, 1, 0}
        });

        //  Tweak this by adding a 1.0 bias for each sample.
        trainingInputs = linalgProvider.getOperations().transpose(trainingInputs);
        linalgProvider.getOperations().prependColumn(trainingInputs, linalgProvider.vectorOf(1.0, trainingInputs.rows()));
        trainingInputs = linalgProvider.getOperations().transpose(trainingInputs);


        trainingOutputs = linalgProvider.getMatrix(new double[][]{new double[]{0,1,1,0}});

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


        System.out.println("Training Samples:  \n" + trainingInputs);


        Matrix z_2Matrix = linalgProvider.getOperations().matrixMatrixProduct(theta1, trainingInputs);
        System.out.println("Z Layer 2:\n"+z_2Matrix);

        Matrix activationsLayer2 = linalgProvider.getOperations().function(z_2Matrix, (entry)->
            1.0 / (1.0 + Math.exp(-1.0 * entry))
        );
        System.out.println("ACTIVATIONS:\n"+activationsLayer2);
    }

    @Test
    public void shouldComputeActivationsOfOutputLayer(){
        Matrix z_2Matrix = linalgProvider.getOperations().matrixMatrixProduct(theta1, trainingInputs);

        Matrix activationsLayer2 = linalgProvider.getOperations().function(z_2Matrix, (entry)->
                1.0 / (1.0 + Math.exp(-1.0 * entry))
        );

        //  Add bias for the activations of the output layer
        activationsLayer2 = linalgProvider.getOperations().transpose(activationsLayer2);
        linalgProvider.getOperations().prependColumn(activationsLayer2, linalgProvider.vectorOf(1.0, activationsLayer2.rows()));
        activationsLayer2 = linalgProvider.getOperations().transpose(activationsLayer2);

        System.out.println(activationsLayer2);

        Matrix z_OutMatrix = linalgProvider.getOperations().matrixMatrixProduct(theta2, activationsLayer2);
        Matrix activationsOutput = linalgProvider.getOperations().function(z_OutMatrix, (entry)->
                1.0 / (1.0 + Math.exp(-1.0 * entry))
        );
        System.out.println(activationsOutput);
    }

    @Test
    public void shouldComputeOutputDeltas(){
        Matrix z_2Matrix = linalgProvider.getOperations().matrixMatrixProduct(theta1, trainingInputs);

        Matrix activationsLayer2 = linalgProvider.getOperations().function(z_2Matrix, (entry)->
                1.0 / (1.0 + Math.exp(-1.0 * entry))
        );

        //  Add bias for the activations of the output layer
        activationsLayer2 = linalgProvider.getOperations().transpose(activationsLayer2);
        linalgProvider.getOperations().prependColumn(activationsLayer2, linalgProvider.vectorOf(1.0, activationsLayer2.rows()));
        activationsLayer2 = linalgProvider.getOperations().transpose(activationsLayer2);

        System.out.println(activationsLayer2);

        Matrix z_OutMatrix = linalgProvider.getOperations().matrixMatrixProduct(theta2, activationsLayer2);
        Matrix hThetaXMatrix = linalgProvider.getOperations().function(z_OutMatrix, (entry)->
                1.0 / (1.0 + Math.exp(-1.0 * entry))
        );
        System.out.println("HTHeta:"+hThetaXMatrix);

        Vector[] yVectors = linalgProvider.toColumnVectors(trainingOutputs);
        Vector[] hThetaVectors = linalgProvider.toColumnVectors(hThetaXMatrix);

        Vector[] deltaVectors = new Vector[yVectors.length];
        for(int i=0; i<yVectors.length; i++){
            deltaVectors[i] = linalgProvider.getOperations().subtract(hThetaVectors[i], yVectors[i]);
        }

        System.out.println("Expected Outputs:"+trainingOutputs);

        Matrix outputDeltaMatrix = linalgProvider.fromVectors(deltaVectors);
        System.out.println("outputDeltas:"+outputDeltaMatrix);
    }


}
