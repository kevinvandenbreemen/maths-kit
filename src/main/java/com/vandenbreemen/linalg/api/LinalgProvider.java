package com.vandenbreemen.linalg.api;

public interface LinalgProvider {

    /**
     * Produce a matrix of values.length rows, and values[0[.length columns
     * @param values
     * @return
     */
    Matrix getMatrix(double[][] values);

    /**
     * Produce a matrix with the given number of rows and columns
     * @param rows
     * @param columns
     * @return
     */
    Matrix getMatrix(int rows, int columns);

}
