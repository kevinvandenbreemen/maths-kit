package com.vandenbreemen.linalg.api;

public interface LinalgOperations {
    Vector hadamard(Vector v1, Vector v2);

    Vector matrixVectorProduct(Matrix m, Vector v);
}
