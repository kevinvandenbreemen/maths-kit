package com.vandenbreemen.linalg.impl;

import com.vandenbreemen.linalg.api.Matrix;
import com.vandenbreemen.linalg.api.Vector;
import org.junit.Before;
import org.junit.Test;

import static junit.framework.TestCase.assertEquals;


public class LinalgProviderImplTest {

    private LinalgProviderImpl linalgProvider;

    @Before
    public void setup(){
        linalgProvider = new LinalgProviderImpl();
    }

    @Test
    public void shouldCreateNewMatrixUsingArrayOfDoubles(){
        //  Act
        Matrix matrix = linalgProvider.getMatrix(new double[][]{
                {0,1},{1,0}
        });

        //  Assert
        assertEquals(2, matrix.rows());
        assertEquals(2, matrix.cols());
    }

    @Test
    public void shouldCreateNewMatrixUsingDimensions(){
        //  Act
        Matrix matrix = linalgProvider.getMatrix(2, 3);

        //  Assert
        assertEquals(2, matrix.rows());
        assertEquals(3, matrix.cols());
    }

    @Test
    public void shouldCreateNewVector(){
        //  Act

        Vector vector = linalgProvider.getVector(new double[]{
                12.2, 6
        });

        //  Assert
        assertEquals("Size", vector.length(), 2);
        assertEquals("Entries", 12.2, vector.entry(0));
        assertEquals("Entries", 6.0, vector.entry(1));
    }

    @Test
    public void shouldCopyVector(){
        //  Arrange
        Vector vector = linalgProvider.getVector(new double[]{
                12.2, 6
        });

        //  Act
        Vector copy = linalgProvider.copyVector(vector);

        //  Assert
        assertEquals("Size", copy.length(), 2);
        assertEquals("Entries", 12.2, copy.entry(0));
        assertEquals("Entries", 6.0, copy.entry(1));
    }

}