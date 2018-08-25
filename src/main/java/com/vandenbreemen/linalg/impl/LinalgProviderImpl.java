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
    public Matrix matrixOf(int rows, int columns, double value) {
        Matrix ret = new MatrixImpl(rows, columns);
        for(int j=0; j<columns; j++){
            for(int i = 0; i<rows; i++){
                ret.set(i, j, value);
            }
        }

        return ret;
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

    @Override
    public Matrix fromVectors(Vector... vectors) {
        double[][] result = new double[vectors[0].length()][vectors.length];
        for(int j=0; j<vectors.length; j++) {
            for (int i = 0; i < vectors[0].length(); i++) {
                result[i][j] = vectors[j].entry(i);
            }
        }
        return new MatrixImpl(result);
    }

    @Override
    public Vector[] toColumnVectors(Matrix matrix) {
        Vector[] ret = new Vector[matrix.cols()];
        for(int i=0; i<matrix.cols(); i++){
            ret[i] = matrix.columnVector(i);
        }
        return ret;
    }

    @Override
    public double[] unroll(Matrix m) {
        double[] ret = new double[m.rows() * m.cols()];

        int index = 0;
        for(int i=0; i<m.cols(); i++){
            for (int j=0; j<m.rows(); j++){
                ret[index] = m.get(j, i);
                index ++;
            }
        }
        return ret;
    }
}
