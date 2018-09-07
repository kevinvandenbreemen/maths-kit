package com.vandenbreemen.linalg;

import org.junit.Test;

import javax.imageio.ImageIO;

public class ImageToMatrixTest {

    @Test
    public void shouldLoadImage() throws Exception{
        ImageIO.read(getClass().getResourceAsStream("/test.png"));
    }

}
