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
}
