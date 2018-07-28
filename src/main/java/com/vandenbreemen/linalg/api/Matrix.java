package com.vandenbreemen.linalg.api;

/**
 * A matrix of some kind
 */
public interface Matrix {


    int rows();

    int cols();

    double get(int row, int col);

    void set(int row, int col, double value);

    double[] matrixVectorProduct(double[] vector);
}
