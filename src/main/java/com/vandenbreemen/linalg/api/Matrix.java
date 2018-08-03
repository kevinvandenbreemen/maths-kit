package com.vandenbreemen.linalg.api;

/**
 * A matrix of some kind
 */
public interface Matrix {


    int rows();

    int cols();

    double get(int row, int col);

    void set(int row, int col, double value);

    default String asString(){
        StringBuilder bld = new StringBuilder(getClass().getSimpleName()).append(":\n");
        for(int i=0; i<rows(); i++){
            bld.append("[ ");
            for(int j=0; j<cols(); j++){
                bld.append(get(i, j)).append("\t");
            }
            bld.append("]\n");
        }
        return bld.toString();
    }
}
