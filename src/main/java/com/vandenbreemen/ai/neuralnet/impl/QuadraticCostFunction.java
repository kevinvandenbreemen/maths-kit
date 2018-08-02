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
    public double cost(TrainingExample... examples) {
        double sum = 0.0;
        for(TrainingExample example: examples){
            Vector difference = linalgProvider.getOperations().subtract(example.getOutput(), example.getActualOutput());
            double magnitude = linalgProvider.getOperations().norm(difference);
            sum += magnitude;
        }

        return (1./(2.*examples.length)) * sum;
    }
}
