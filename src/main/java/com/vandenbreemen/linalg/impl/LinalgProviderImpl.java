package com.vandenbreemen.linalg.impl;

import com.vandenbreemen.linalg.api.LinalgOperations;
import com.vandenbreemen.linalg.api.LinalgProvider;
import com.vandenbreemen.linalg.api.Matrix;
import com.vandenbreemen.linalg.api.Vector;

public class LinalgProviderImpl implements LinalgProvider {
    @Override
    public Matrix getMatrix(double[][] values) {
        //  Verify all rows are same length:
        int len = values[0].length;
        for(int i=1; i<values.length; i++){
            if(values[i].length != len){
                throw new RuntimeException("Raw data cannot be a ragged array.  Row "+i+" is missing at least one entry");
            }
        }
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

    @Override
    public Vector vectorOf(double value, int numRows) {
        double[] values = new double[numRows];
        for(int i=0; i<numRows; i++){
            values[i] = value;
        }
        return new VectorImpl(values);
    }
}
