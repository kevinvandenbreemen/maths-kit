package com.vandenbreemen.linalg.impl;

import com.vandenbreemen.linalg.api.Matrix;
import com.vandenbreemen.linalg.api.Vector;

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
    public Vector columnVector(int columnIndex) {
        double[] data = new double[rows()];
        for(int j=0; j<rows(); j++){
            data[j] = this.data[j][columnIndex];
        }
        return new VectorImpl(data);
    }
}
