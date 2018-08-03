package com.vandenbreemen.ai.neuralnet.impl;

import com.vandenbreemen.ai.neuralnet.api.CostFunction;
import com.vandenbreemen.ai.neuralnet.api.TrainingExample;
import com.vandenbreemen.linalg.api.LinalgProvider;
import com.vandenbreemen.linalg.api.Vector;

public class QuadraticCostFunction implements CostFunction {

    private LinalgProvider linalgProvider;

    public QuadraticCostFunction(LinalgProvider linalgProvider) {
        this.linalgProvider = linalgProvider;
    }

    @Override
    public double averageCost(TrainingExample... examples) {
        double sum = 0.0;
        for(TrainingExample example: examples){
            sum += cost(example);
        }

        return (1./(examples.length)) * sum;
    }

    @Override
    public double cost(TrainingExample example) {
        Vector difference = linalgProvider.getOperations().subtract(example.getExpectedOutput(), example.getActualOutput());
        double magnitude = linalgProvider.getOperations().norm(difference);
        return 0.5 * (magnitude*magnitude);
    }

    @Override
    public Vector gradient(TrainingExample example) {
        double[] result = new double[example.getExpectedOutput().length()];
        for(int i=0; i<result.length; i++){
            result[i] = example.getActualOutput().entry(i) - example.getExpectedOutput().entry(i);
        }
        return linalgProvider.getVector(result);
    }
}
