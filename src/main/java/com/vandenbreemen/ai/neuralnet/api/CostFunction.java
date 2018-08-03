package com.vandenbreemen.ai.neuralnet.api;

import com.vandenbreemen.linalg.api.Vector;

public interface CostFunction {

    double averageCost(TrainingExample...examples);

    double cost(TrainingExample example);

    Vector gradient(TrainingExample example);

}
