package com.vandenbreemen.linalg.impl;

import com.vandenbreemen.linalg.api.Matrix;
import com.vandenbreemen.linalg.api.Vector;
import org.junit.Test;

import static junit.framework.TestCase.assertEquals;


public class LinalgProviderImplTest {

    @Test
    public void shouldCreateNewMatrixUsingArrayOfDoubles(){
        //  Act
        Matrix matrix = new LinalgProviderImpl().getMatrix(new double[][]{
                {0,1},{1,0}
        });

        //  Assert
        assertEquals(2, matrix.rows());
        assertEquals(2, matrix.cols());
    }

    @Test
    public void shouldCreateNewMatrixUsingDimensions(){
        //  Act
        Matrix matrix = new LinalgProviderImpl().getMatrix(2, 3);

        //  Assert
        assertEquals(2, matrix.rows());
        assertEquals(3, matrix.cols());
    }

    @Test
    public void shouldCreateNewVector(){
        //  Act
        Vector vector = new LinalgProviderImpl().getVector(new double[]{
                12.2, 6
        });

        //  Assert
        assertEquals("Size", vector.length(), 2);
        assertEquals("Entries", 12.2, vector.entry(0));
        assertEquals("Entries", 6.0, vector.entry(1));
    }

}