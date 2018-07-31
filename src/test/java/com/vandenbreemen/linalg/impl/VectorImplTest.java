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

}
