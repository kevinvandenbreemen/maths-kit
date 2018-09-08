package com.vandenbreemen.linalg;

import com.vandenbreemen.linalg.api.LinalgProvider;
import com.vandenbreemen.linalg.api.Matrix;
import com.vandenbreemen.linalg.api.Vector;
import com.vandenbreemen.linalg.impl.LinalgProviderImpl;
import org.junit.Before;
import org.junit.Test;

import static junit.framework.TestCase.assertEquals;
import static junit.framework.TestCase.assertNotNull;

public class ImageToMatrixTest {

    private ImageToMatrix imageToMatrix;

    private LinalgProvider provider;

    @Before
    public void setup(){
        this.provider = new LinalgProviderImpl();
        this.imageToMatrix = new ImageToMatrix(provider);
    }

    @Test
    public void shouldLoadImage() throws Exception{
        imageToMatrix.getBufferedImage(getClass().getResourceAsStream("/test.png"));
    }

    @Test
    public void shouldExtractRGBArray(){
        double[][] rgbArray = imageToMatrix.getRBBArray(getClass().getResourceAsStream("/test.png"));
        assertNotNull(rgbArray);

        StringBuilder bld = new StringBuilder();
        int index = 0;
        for(int i=0; i<10; i++){
            for(int j=0; j<10; j++){
                bld.append(rgbArray[0][index]).append("\t");
                index++;
            }
            bld.append("\n");

        }

        System.out.println(bld);

        assertEquals("Pixel Dim", 100, rgbArray[0].length);
        assertEquals(0.0, rgbArray[0][91]);
        assertEquals(255.0, rgbArray[0][0]);
        assertEquals(255.0, rgbArray[1][0]);
        assertEquals(255.0, rgbArray[2][0]);
    }

    @Test
    public void shouldConvertToGrayscale(){
        Vector grayScaleVector = imageToMatrix.getGrayscaleVector(getClass().getResourceAsStream("/test.png"));
        Matrix testMatrix = provider.reshape(grayScaleVector, 10,10);

        System.out.println(testMatrix);

    }

}
