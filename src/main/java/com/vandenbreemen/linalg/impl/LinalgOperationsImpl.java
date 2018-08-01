package com.vandenbreemen.linalg.impl;

import com.vandenbreemen.linalg.api.LinalgOperations;
import com.vandenbreemen.linalg.api.Matrix;
import com.vandenbreemen.linalg.api.Vector;

public class LinalgOperationsImpl implements LinalgOperations {
    @Override
    public Vector hadamard(Vector v1, Vector v2) {
        if(v1.length() != v2.length()){
            throw new RuntimeException("Vectors have incompatible lengths - "+v1.length() + " vs "+v2.length());
        }
        double[] product = new double[v1.length()];
        for(int i=0; i<v1.length(); i++){
            product[i] = v1.entry(i) * v2.entry(i);
        }

        return new VectorImpl(product);
    }

    @Override
    public Vector matrixVectorProduct(Matrix m, Vector v) {
        if(v.length() != m.cols()){
            throw new RuntimeException("Vector must have same number of entries as columns ("+m.cols()+")");
        }

        double[] result = new double[m.rows()];
        for(int i=0; i<m.rows(); i++){
            double sum = 0.;
            for(int j=0; j<m.cols(); j++){
                sum += m.get(i,j) * v.entry(j);
            }
            result[i] = sum;
        }

        return new VectorImpl(result);
    }
}
