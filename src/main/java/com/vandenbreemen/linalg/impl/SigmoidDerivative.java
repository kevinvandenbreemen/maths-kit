package com.vandenbreemen.linalg.impl;

import com.vandenbreemen.linalg.api.VectorFunction;

/**
 * σ(x)⋅(1−σ(x)).
 */
public class SigmoidDerivative implements VectorFunction {

    private SigmoidFunction sigmoidFunction;

    public SigmoidDerivative(SigmoidFunction sigmoidFunction) {
        this.sigmoidFunction = sigmoidFunction;
    }

    @Override
    public double operate(double entry) {
        double sigmoid = sigmoidFunction.operate(entry);
        return sigmoid * (1 - sigmoid);
    }
}
