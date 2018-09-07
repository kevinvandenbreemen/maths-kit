package com.vandenbreemen.linalg;

import com.vandenbreemen.linalg.impl.LinalgProviderImpl;
import org.junit.Before;
import org.junit.Test;

import javax.imageio.ImageIO;

import static junit.framework.TestCase.assertEquals;
import static junit.framework.TestCase.assertNotNull;

public class ImageToMatrixTest {

    private ImageToMatrix imageToMatrix;

    @Before
    public void setup(){
        this.imageToMatrix = new ImageToMatrix(new LinalgProviderImpl());
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
    }

}
