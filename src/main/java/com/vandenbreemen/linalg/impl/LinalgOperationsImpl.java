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
    public Matrix hadamard(Matrix m1, Matrix m2) {
        if(m1.cols() != m2.cols() || m1.rows() != m2.rows()){
            throw new RuntimeException("Cannot compute hadamard for matrices of differing size ("+m1.rows()+" x "+m1.cols()+") vs ("+m2.rows()+" x "+m2.cols()+")");
        }

        double[][] productData = new double[m1.rows()][m1.cols()];
        for(int i=0; i<m1.cols(); i++){
            for(int j=0; j<m1.rows(); j++){
                productData[j][i] = m1.get(j, i) * m2.get(j, i);
            }
        }

        return new MatrixImpl(productData);
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

    private String dimensionDifference(Matrix m1, Matrix m2){
        return m1.rows()+" x "+m1.cols() + " vs "+m2.rows()+" x "+m2.cols();
    }

    @Override
    public Matrix add(Matrix m1, Matrix m2) {
        if(m1.cols() != m2.cols()||m1.rows() != m2.rows()){
            throw new RuntimeException("Cannot add matrices of differing dimensions ("+dimensionDifference(m1, m2)+")");
        }

        double[][] result = new double[m1.rows()][m1.cols()];
        for(int j=0; j<m1.cols(); j++){
            for(int i=0; i<m1.rows(); i++){
                result[i][j] = m1.get(i,j)+m2.get(i,j);
            }
        }

        return new MatrixImpl(result);
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
    public Matrix function(Matrix matrix, VectorFunction function) {
        double[][] result = new double[matrix.rows()][matrix.cols()];
        for(int i=0; i<matrix.cols(); i++){
            for(int j=0; j<matrix.rows(); j++){
                result[j][i] = function.operate(matrix.get(j, i));
            }
        }
        return new MatrixImpl(result);
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
    public double sum(Vector vector) {
        double sum = 0.0;
        for(int i=0; i<vector.length(); i++){
            sum += vector.entry(i);
        }
        return sum;
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

    @Override
    public Matrix matrixMatrixProduct(Matrix m, Matrix n) {
        if(m.cols() != n.rows()){
            throw new RuntimeException("Cannot multiply "+m.rows()+" x "+m.cols()+" matrix by "+n.rows() + " x "+n.cols() + " matrix");
        }

        double[][] result = new double[m.rows()][n.cols()];

        Vector columnVector;
        Vector vProduct;

        for(int i=0; i<n.cols(); i++){
            columnVector = n.columnVector(i);
            vProduct = matrixVectorProduct(m, columnVector);

            for (int j = 0; j < vProduct.length(); j++) {
                result[j][i] = vProduct.entry(j);
            }

        }

        return new MatrixImpl(result);
    }

    @Override
    public void prependColumn(Matrix m, Vector vector) {
        MatrixImpl impl = (MatrixImpl) m;
        double[][] updated = new double[m.rows()][m.cols()+1];
        for(int j=0; j<m.rows(); j++){
            updated[j][0] = vector.entry(j);
        }
        for(int i = 0; i<m.cols(); i++){
            for(int j=0; j<m.rows(); j++){
                updated[j][i+1] = m.get(j, i);
            }
        }

        ((MatrixImpl) m).data = updated;
    }

    @Override
    public Vector prependEntry(Vector vector, double entry) {
        double[] values = new double[vector.length()+1];
        values[0] = entry;
        for(int i=0; i<vector.length(); i++){
            values[i+1] = vector.entry(i);
        }
        return new VectorImpl(values);
    }

    @Override
    public Vector vectorScalarProduct(double scalar, Vector vector) {
        double[] result = new double[vector.length()];
        for(int i=0; i<vector.length(); i++){
            result[i] = scalar * vector.entry(i);
        }
        return new VectorImpl(result);
    }

    @Override
    public Matrix copy(Matrix matrix) {
        return transpose(transpose(matrix));
    }

    @Override
    public Matrix subMatrixFromRow(Matrix matrix, int rowIndex) {
        double[][] result = new double[matrix.rows()-1][matrix.cols()];
        for(int i=0; i<matrix.cols(); i++){
            for(int j=0; j<matrix.rows()-rowIndex; j++){
                result[j][i] = matrix.get(j+rowIndex, i);
            }
        }
        return new MatrixImpl(result);
    }
}
