package com.vandenbreemen.linalg;

import com.vandenbreemen.linalg.api.LinalgProvider;
import com.vandenbreemen.linalg.api.Matrix;
import com.vandenbreemen.linalg.api.Vector;
import com.vandenbreemen.linalg.impl.LinalgProviderImpl;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.InputStream;

public class ImageToMatrix {

    private LinalgProvider linalgProvider;

    /**
     * Transform for conversion to grayscale
     */
    private Matrix linearTransformMatrix;

    public ImageToMatrix(LinalgProvider linalgProvider) {
        this.linalgProvider = linalgProvider;
        this.linearTransformMatrix = linalgProvider.getMatrix(new double[][]{
                new double[]{0.2126, 0.7152, 0.0722}
        });
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

    //  Grayscale conversion as per https://stackoverflow.com/a/17619494/2328196
    public Vector getGrayscaleVector(InputStream resourceAsStream) {
        Matrix imageMatrix = linalgProvider.getMatrix(getRBBArray(resourceAsStream));
        imageMatrix = linalgProvider.getOperations().function(imageMatrix, (e)->e/255.0);

        Matrix linearMatrix = linalgProvider.getOperations().matrixMatrixProduct(linearTransformMatrix, imageMatrix);

        linearMatrix = linalgProvider.getOperations().function(linearMatrix, e->{
            if(e <= 0.0031308){
                return 12.92 * e;
            }
            else{
                return (1.055 * (Math.pow(e, (1.0/2.5)))) - 0.055;
            }
        });

        return linalgProvider.getVector(linalgProvider.unroll(linearMatrix));

    }
}
