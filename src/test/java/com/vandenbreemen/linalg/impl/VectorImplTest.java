package com.vandenbreemen.linalg.impl;

import com.vandenbreemen.linalg.api.LinalgProvider;
import com.vandenbreemen.linalg.api.Vector;
import org.junit.Before;
import org.junit.Test;

import static junit.framework.TestCase.assertEquals;

public class VectorImplTest {

    private LinalgProvider provider;

    @Before
    public void setup(){
        this.provider = new LinalgProviderImpl();
    }

    @Test
    public void shouldComputeHadamardProduct(){
        //  Arrange
        Vector v1 = provider.getVector(new double[]{2,2});
        Vector v2 = provider.getVector(new double[]{2,3});

        //  Act
        Vector hadamard = provider.getOperations().hadamard(v1, v2);

        //  Assert
        assertEquals("First Entry", 4.0, hadamard.entry(0));
        assertEquals("Second Entry", 6.0, hadamard.entry(1));
    }

    @Test
    public void shouldAddVectors(){

        //  Arrange
        Vector v1 = provider.getVector(new double[]{2,3});
        Vector v2 = provider.getVector(new double[]{3,3});

        //  Act
        Vector sum = provider.getOperations().add(v1, v2);

        //  Assert
        assertEquals(5.0, sum.entry(0));
        assertEquals(6.0, sum.entry(1));
    }

    @Test
    public void shouldOperateOnVector(){

        //  Arrange
        Vector vector = provider.getVector(new double[]{3, 5, 6});

        //  Act
        Vector operated = provider.getOperations().function(vector, (e)->e*2);

        //  Assert
        assertEquals(6.0, operated.entry(0));
        assertEquals(10.0, operated.entry(1));
        assertEquals(12.0, operated.entry(2));

    }

    @Test
    public void shouldComputeVectorNorm(){
        //  Arrange
        Vector vector = provider.getVector(new double[]{3, 4});

        //  Act
        double norm = provider.getOperations().norm(vector);

        //  Assert
        assertEquals(5.0, norm);
    }

}
