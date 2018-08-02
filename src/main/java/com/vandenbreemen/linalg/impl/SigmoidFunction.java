package com.vandenbreemen.linalg.impl;

import com.vandenbreemen.linalg.api.VectorFunction;

public class SigmoidFunction implements VectorFunction {
    @Override
    public double operate(double entry) {
        return 1.0 / (1 + Math.exp(-1 * entry));
    }
}
