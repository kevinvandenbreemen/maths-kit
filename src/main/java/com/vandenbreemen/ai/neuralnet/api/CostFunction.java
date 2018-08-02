package com.vandenbreemen.ai.neuralnet.api;

@FunctionalInterface
public interface CostFunction {

    double cost(TrainingExample...examples);

}
