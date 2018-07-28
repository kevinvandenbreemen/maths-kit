package com.vandenbreemen.linalg.impl;

import com.vandenbreemen.linalg.api.Matrix;
import org.junit.Test;

import static junit.framework.TestCase.assertEquals;
import static org.junit.Assert.assertArrayEquals;


public class MatrixImplTest {

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

        //  Act
        double[] product = m.matrixVectorProduct(new double[]{
                2, 3, 7.9
        });

        //  Assert
        double[] expected = new double[]{
                -18.6, 101.3
        };
        assertArrayEquals(expected, product, 0.00001);
    }

}