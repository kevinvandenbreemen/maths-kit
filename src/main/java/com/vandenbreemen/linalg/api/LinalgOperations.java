package com.vandenbreemen.linalg.api;

public interface LinalgOperations {
    Vector hadamard(Vector v1, Vector v2);

    Vector matrixVectorProduct(Matrix m, Vector v);

    Vector add(Vector v1, Vector v2);

    Matrix randomEntries(Matrix matrix);

    Vector randomEntries(Vector vector);

    Vector function(Vector vector, VectorFunction function);

    Matrix function(Matrix matrix, VectorFunction function);

    double norm(Vector vector);

    Vector subtract(Vector v1, Vector v2);

    Matrix transpose(Matrix m);

    Matrix matrixMatrixProduct(Matrix m, Matrix n);

    void prependColumn(Matrix m, Vector vector);

    Vector vectorScalarProduct(double scalar, Vector vector);
}
