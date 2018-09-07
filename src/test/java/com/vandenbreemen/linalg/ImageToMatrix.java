package com.vandenbreemen.linalg;

import com.vandenbreemen.linalg.api.LinalgProvider;
import com.vandenbreemen.linalg.impl.LinalgProviderImpl;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.InputStream;

public class ImageToMatrix {

    private LinalgProvider linalgProvider;

    public ImageToMatrix(LinalgProvider linalgProvider) {
        this.linalgProvider = linalgProvider;
    }

    protected BufferedImage getBufferedImage(InputStream inputStream) {
        try{
            return ImageIO.read(inputStream);
        }
        catch(Exception ex){
            ex.printStackTrace();
            throw new RuntimeException("Failed to load image", ex);
        }
    }

    /**
     * Converts the image into a strip of RGB pixel values
     * @param resourceAsStream
     * @return
     */
    public double[][] getRBBArray(InputStream resourceAsStream) {
        BufferedImage image = getBufferedImage(resourceAsStream);
        int w = image.getWidth();
        int h = image.getHeight();

        double[][] ret = new double[3][w*h];

        int index = 0;

        for (int i = 0; i < h; i++) {
            for (int j = 0; j < w; j++) {
                float[] rgba = new float[4];
                image.getRaster().getPixel(j, i, rgba);

                for(int k=0; k<3; k++){
                    ret[k][index] = rgba[k];
                }

                index ++;
            }
        }

        return ret;
    }
}
