package com.vandenbreemen.linalg.impl;

import com.vandenbreemen.linalg.api.Matrix;
import org.junit.Test;

import static org.junit.Assert.*;

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

}