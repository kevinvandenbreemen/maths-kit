package com.vandenbreemen.linalg.impl;

import com.vandenbreemen.linalg.api.LinalgProvider;
import com.vandenbreemen.linalg.api.Matrix;
import com.vandenbreemen.linalg.api.Vector;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static junit.framework.TestCase.assertEquals;
import static junit.framework.TestCase.assertTrue;
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

    @Test
    public void shouldMultiplyTwoMatrices(){
        Matrix m = provider.getMatrix(new double[][]{
                new double[]{1.0, 0.9},
                new double[]{0.1, 0.2},
                new double[]{1.,1.}
        });
        Matrix n = provider.getMatrix(new double[][]{
                new double[]{1.0,1.0},
                new double[]{2.0,2.0}
        });

        Matrix product = provider.getOperations().matrixMatrixProduct(m, n);
        assertEquals(3, product.rows());
        assertEquals(2, product.cols());

    }

    @Test
    public void shouldGetColumnVector(){
        Matrix m = provider.getMatrix(new double[][]{
                new double[]{1.0, 0.9},
                new double[]{0.1, 0.2},
                new double[]{1.,1.}
        });

        Vector column = m.columnVector(0);
        assertEquals(3, column.length());
        assertEquals(1.0, column.entry(0));
        assertEquals(0.1, column.entry(1));
        assertEquals(1.0, column.entry(2));
    }

    @Test
    public void shouldAddVectorToMatrix(){
        Matrix m = provider.getMatrix(new double[][]{
                new double[]{1.0, 0.9},
                new double[]{0.1, 0.2},
                new double[]{1.,1.}
        });

        System.out.println(m);

        Vector vector = provider.getVector(new double[]{1.0,1.0,1.0});
        provider.getOperations().prependColumn(m, vector);
        System.out.println(m);

        assertEquals(3, m.cols());

        assertEquals(1.0, m.get(0,0));
        assertEquals(1.0, m.get(1,0));
        assertEquals(1.0, m.get(2,0));
        assertEquals(1.0, m.get(0,1));
        assertEquals(0.1, m.get(1,1));
        assertEquals(1.0, m.get(2,1));
        assertEquals(0.9, m.get(0,2));
        assertEquals(0.2, m.get(1,2));
        assertEquals(1.0, m.get(2,2));

    }

    @Test
    public void shouldOperateOnEntries(){
        Matrix m = provider.getMatrix(new double[][]{
                new double[]{1.0, 2.0},
                new double[]{3.0, 4.0}
        });

        Matrix up = provider.getOperations().function(m, e->e*2);
        assertEquals(2.0, up.get(0,0));
        assertEquals(6.0, up.get(1,0));
        assertEquals(4.0, up.get(0,1));
        assertEquals(8.0, up.get(1,1));
    }

    @Test
    public void shouldUnrollAndRullUpColumnVectors(){
        Matrix m = provider.getMatrix(new double[][]{
                new double[]{1.0, 2.0},
                new double[]{3.0, 4.0}
        });

        Vector[] vectors = provider.toColumnVectors(m);
        assertEquals(2, vectors.length);

        assertEquals(1.0,vectors[0].entry(0));
        assertEquals(3.0,vectors[0].entry(1));
        assertEquals(2.0,vectors[1].entry(0));
        assertEquals(4.0,vectors[1].entry(1));

        Matrix n = provider.fromVectors(vectors);
        assertEquals(1.0, n.get(0,0));
        assertEquals(3.0, n.get(1,0));
        assertEquals(2.0, n.get(0,1));
        assertEquals(4.0, n.get(1,1));
    }

    @Test
    public void shouldUnrollMatrix(){
        Matrix m = provider.getMatrix(new double[][]{
                new double[]{1.0, 2.0},
                new double[]{3.0, 4.0}
        });

        double[] values = provider.unroll(m);

        assertEquals(4, values.length);
        assertTrue(Arrays.stream(values).filter(d->d == 1.0).findFirst().isPresent());
        assertTrue(Arrays.stream(values).filter(d->d == 2.0).findFirst().isPresent());
        assertTrue(Arrays.stream(values).filter(d->d == 3.0).findFirst().isPresent());
        assertTrue(Arrays.stream(values).filter(d->d == 4.0).findFirst().isPresent());
    }

    @Test
    public void shouldHadamardMatrices(){
        Matrix m = provider.getMatrix(new double[][]{
                new double[]{1.0, 2.0},
                new double[]{3.0, 4.0}
        });
        Matrix n = provider.getMatrix(new double[][]{
                new double[]{2.0, 2.0},
                new double[]{2.0, 2.0}
        });

        Matrix up = provider.getOperations().hadamard(m, n);
        assertEquals(2.0, up.get(0,0));
        assertEquals(6.0, up.get(1,0));
        assertEquals(4.0, up.get(0,1));
        assertEquals(8.0, up.get(1,1));
    }

}