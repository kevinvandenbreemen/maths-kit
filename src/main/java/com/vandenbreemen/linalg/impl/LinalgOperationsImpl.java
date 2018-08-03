package com.vandenbreemen.linalg.impl;

import com.vandenbreemen.linalg.api.LinalgOperations;
import com.vandenbreemen.linalg.api.Matrix;
import com.vandenbreemen.linalg.api.Vector;
import com.vandenbreemen.linalg.api.VectorFunction;

import java.util.Random;

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

    @Override
    public Vector add(Vector v1, Vector v2) {
        if(v1.length() != v2.length()){
            throw new RuntimeException("both vectors must have same number of entries ("+v1.length()+")");
        }

        double[] result = new double[v1.length()];
        for(int i=0; i<v1.length(); i++){
            result[i] = v1.entry(i)+v2.entry(i);
        }

        return new VectorImpl(result);
    }

    @Override
    public Matrix randomEntries(Matrix matrix) {
        Random random = new Random(System.nanoTime());
        for(int i=0; i<matrix.rows(); i++){
            for(int j=0; j<matrix.cols(); j++){
                matrix.set(i, j, random.nextDouble());
            }
        }
        return matrix;
    }

    @Override
    public Vector randomEntries(Vector vector) {
        Random random = new Random(System.nanoTime());
        double[] values = new double[vector.length()];
        for(int i=0; i< vector.length(); i++){
            values[i] = random.nextDouble();
        }
        return new VectorImpl(values);
    }

    @Override
    public Vector function(Vector vector, VectorFunction function) {
        double[] result = new double[vector.length()];
        for(int i=0; i<vector.length(); i++){
            result[i] = function.operate(vector.entry(i));
        }
        return new VectorImpl(result);
    }

    @Override
    public double norm(Vector vector) {
        double sum = 0.0;
        for(int i=0; i<vector.length(); i++){
            sum += vector.entry(i) * vector.entry(i);
        }

        return Math.sqrt(sum);
    }

    @Override
    public Vector subtract(Vector v1, Vector v2) {
        if(v1.length() != v2.length()){
            throw new RuntimeException("both vectors must have same number of entries ("+v1.length()+")");
        }

        double[] result = new double[v1.length()];
        for(int i=0; i<v1.length(); i++){
            result[i] = v1.entry(i)-v2.entry(i);
        }

        return new VectorImpl(result);
    }

    @Override
    public Matrix transpose(Matrix m) {
        double[][] values = new double[m.cols()][m.rows()];
        for(int i=0; i<m.rows(); i++){
            for(int j=0; j<m.cols(); j++){
                values[j][i] = m.get(i,j);
            }
        }
        return new MatrixImpl(values);
    }
}
