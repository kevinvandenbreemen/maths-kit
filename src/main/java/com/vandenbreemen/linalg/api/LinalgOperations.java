package com.vandenbreemen.linalg.api;

public interface LinalgOperations {
    Vector hadamard(Vector v1, Vector v2);

    Vector matrixVectorProduct(Matrix m, Vector v);

    Vector add(Vector v1, Vector v2);

    Matrix randomEntries(Matrix matrix);

    Vector randomEntries(Vector vector);

    Vector function(Vector vector, VectorFunction function);

    double norm(Vector vector);

    Vector subtract(Vector v1, Vector v2);
}
