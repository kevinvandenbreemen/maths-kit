package com.vandenbreemen.ai.neuralnet.api;

import com.vandenbreemen.ai.neuralnet.impl.NeuralNetLayerImpl;
import com.vandenbreemen.ai.neuralnet.impl.QuadraticCostFunction;
import com.vandenbreemen.linalg.api.LinalgProvider;
import com.vandenbreemen.linalg.api.Matrix;
import com.vandenbreemen.linalg.api.Vector;
import com.vandenbreemen.linalg.impl.LinalgProviderImpl;
import com.vandenbreemen.linalg.impl.SigmoidFunction;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static junit.framework.TestCase.assertEquals;


public class NeuralNetPOCTest {

    private LinalgProvider linalgProvider;

    @Before
    public void setup(){
        this.linalgProvider = new LinalgProviderImpl();
    }

    @Test
    public void shouldGenerateLayerActivations(){

        //  Arrange
        int outputCount = 3;
        Vector inputVector = linalgProvider.getVector(new double[]{1.0, 0.0});
        Vector layerBiasVector = linalgProvider.getVector(new double[outputCount]);
        layerBiasVector = linalgProvider.getOperations().randomEntries(layerBiasVector);
        Matrix weightMatrix = linalgProvider.getMatrix(outputCount, inputVector.length());
        weightMatrix = linalgProvider.getOperations().randomEntries(weightMatrix);

        //  Act
        Vector weightedInputs = linalgProvider.getOperations().matrixVectorProduct(weightMatrix, inputVector);
        weightedInputs = linalgProvider.getOperations().add(weightedInputs, layerBiasVector);

        //  Assert/test
        System.out.println(inputVector);
        System.out.println(weightedInputs);

        System.out.println(linalgProvider.getOperations().function(weightedInputs, new SigmoidFunction()));

        //  If the place hasn't burned down it's been a good day!
    }

    @Test
    public void shouldGenerateLayerActivationsUsingLayerType(){

        //  Arrange
        NeuralNetLayer layer = new NeuralNetLayerImpl(linalgProvider, 2, 3);

        //  Act
        Vector activations = layer.getActivation(linalgProvider.getVector(new double[]{1.0, 0.0}));

        //  Assert/Test
        System.out.println("ACTIVATIONS:  "+activations);
    }

    @Test
    public void shouldGenerateQuadraticCostForSingleTrainingExample(){
        //  Arrange
        Vector expected = linalgProvider.getVector(new double[]{1,0,1});
        Vector actual = linalgProvider.getVector(new double[]{0.89, 0.001, 0.2});
        int numSamples = 1;

        //  Act
        Vector difference = linalgProvider.getOperations().subtract(expected, actual);
        double magnitude = linalgProvider.getOperations().norm(difference);
        magnitude = magnitude*magnitude;    //  Squared
        System.out.println(magnitude);

        //  print
        double cost = (1./(2.*numSamples)) * magnitude;
        System.out.println(cost);
    }

    @Test
    public void shouldComputeGradientAsZeroForQuadraticCostFunctionAtZero(){
        //  Arrange
        TrainingExample example = new TrainingExample(
                linalgProvider.getVector(new double[]{0., 1.}),
                linalgProvider.getVector(new double[]{1., 0.})
        );
        example.setActualOutput(linalgProvider.getVector(new double[]{1., 0.}));

        //  Act
        Vector gradient = new QuadraticCostFunction(linalgProvider).gradient(example);

        //  Assert
        assertEquals(2, gradient.length());
        assertEquals(0., gradient.entry(0));
        assertEquals(0., gradient.entry(1));
    }

    @Test
    public void shouldCalculateGradientForQuadraticCostFunction(){
        //  Arrange
        TrainingExample example = new TrainingExample(
                linalgProvider.getVector(new double[]{0., 1.}),
                linalgProvider.getVector(new double[]{1., 0.})
        );
        example.setActualOutput(linalgProvider.getVector(new double[]{1.1, 0.22}));

        //  Act
        Vector gradient = new QuadraticCostFunction(linalgProvider).gradient(example);

        //  Print
        System.out.println(gradient);
    }

}