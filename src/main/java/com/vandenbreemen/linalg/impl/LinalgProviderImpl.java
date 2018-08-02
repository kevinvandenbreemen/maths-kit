package com.vandenbreemen.linalg.impl;

import com.vandenbreemen.linalg.api.LinalgOperations;
import com.vandenbreemen.linalg.api.LinalgProvider;
import com.vandenbreemen.linalg.api.Matrix;
import com.vandenbreemen.linalg.api.Vector;

public class LinalgProviderImpl implements LinalgProvider {
    @Override
    public Matrix getMatrix(double[][] values) {
        return new MatrixImpl(values);
    }

    @Override
    public Matrix getMatrix(int rows, int columns) {
        return new MatrixImpl(rows, columns);
    }

    @Override
    public Vector getVector(double[] entries) {
        return new VectorImpl(entries);
    }

    @Override
    public Vector copyVector(Vector vector) {
        double[] values = new double[vector.length()];
        for(int i=0; i<vector.length(); i++){
            values[i] = vector.entry(i);
        }
        return new VectorImpl(values);
    }

    @Override
    public LinalgOperations getOperations() {
        return new LinalgOperationsImpl();
    }
}
