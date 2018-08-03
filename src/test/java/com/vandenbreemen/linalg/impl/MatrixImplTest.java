package com.vandenbreemen.linalg.impl;

import com.vandenbreemen.linalg.api.LinalgProvider;
import com.vandenbreemen.linalg.api.Matrix;
import com.vandenbreemen.linalg.api.Vector;
import org.junit.Before;
import org.junit.Test;

import static junit.framework.TestCase.assertEquals;
import static org.junit.Assert.assertArrayEquals;


public class MatrixImplTest {

    private LinalgProvider provider;

    @Before
    public void setup(){
        this.provider = new LinalgProviderImpl();
    }

    @Test
    public void shouldSetEntry(){

        //  Act
        Matrix m = new MatrixImpl(2,3);
        m.set(0, 1, 10.3);

        //  Assert
        assertEquals(10.3, m.get(0,1));

    }

    @Test
    public void shouldMultiplyByVector(){
        //  Arrange
        Matrix m = new MatrixImpl(new double[][]{
                {2,3,-4},
                {11, 8, 7}
        });

        Vector v = provider.getVector(new double[]{
                2, 3, 7.9
        });

        //  Act
        Vector product = provider.getOperations().matrixVectorProduct(m, v);

        //  Assert
        assertEquals("Product", -18.6, product.entry(0), 0.00000001);
        assertEquals("Product", 101.3, product.entry(1), 0.00000001);
    }

    @Test
    public void shouldTranspose(){
        //  Arrange
        Matrix m = provider.getMatrix(new double[][]{
                {1,2,3},
                {4,5,6}
        });

        //  Act
        Matrix transpose = provider.getOperations().transpose(m);

        //  Assert
        assertEquals(3, transpose.rows());
        assertEquals(2, transpose.cols());
        assertEquals(1., transpose.get(0,0));
        assertEquals(4., transpose.get(0,1));
        assertEquals(2., transpose.get(1,0));
        assertEquals(5., transpose.get(1,1));
        assertEquals(3., transpose.get(2,0));
        assertEquals(6., transpose.get(2,1));

    }

}