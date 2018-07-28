package com.vandenbreemen.linalg.impl;

import com.vandenbreemen.linalg.api.Matrix;

public class MatrixImpl implements Matrix {

    /**
     * <pre>
     * data.length = rows
     * data[0].length = columns
     * </pre>
     */
    private double[][] data;

    MatrixImpl(double[][] values) {
        this.data = values;
    }

    MatrixImpl(int rows, int cols){
        this.data = new double[rows][cols];
    }

    @Override
    public int rows() {
        return data.length;
    }

    @Override
    public int cols() {
        return data[0].length;
    }

    @Override
    public double get(int row, int col) {
        return data[row][col];
    }

    @Override
    public void set(int row, int col, double value) {
        this.data[row][col] = value;
    }

    @Override
    public double[] matrixVectorProduct(double[] vector) {
        if(vector.length != cols()){
            throw new RuntimeException("Vector must have same number of entries as columns ("+cols()+")");
        }

        double[] result = new double[rows()];
        for(int i=0; i<rows(); i++){
            double sum = 0.;
            for(int j=0; j<cols(); j++){
                sum += data[i][j] * vector[j];
            }
            result[i] = sum;
        }

        return result;
    }
}
