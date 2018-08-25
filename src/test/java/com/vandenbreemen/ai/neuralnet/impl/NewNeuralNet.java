package com.vandenbreemen.ai.neuralnet.impl;

import com.vandenbreemen.linalg.api.LinalgProvider;
import com.vandenbreemen.linalg.api.Matrix;
import com.vandenbreemen.linalg.api.Vector;
import com.vandenbreemen.linalg.api.VectorFunction;

import java.util.ArrayList;
import java.util.List;

public class NewNeuralNet {

    private VectorFunction activationFunction;

    private List<Matrix> thetaMatrices;

    private LinalgProvider provider;

    public NewNeuralNet(LinalgProvider provider) {
        this.provider = provider;
        this.thetaMatrices = new ArrayList<>();
        this.activationFunction = (entry)-> 1.0 / (1.0 + Math.exp(-1.0 * entry));
    }

    public void addLayer(int numInputs, int numOutputs){
        Matrix matrix = provider.getMatrix(numOutputs, numInputs+1);    //  Add 1 to account for the bias unit
        matrix = provider.getOperations().randomEntries(matrix);
        thetaMatrices.add(matrix);
    }

    /**
     *
     * @param x
     */
    public Vector compute_hTheta(Vector x){

        Vector input = x;

        for(Matrix theta : thetaMatrices){
            Vector withBias = provider.getOperations().prependEntry(input, 1.0);
            Vector z_nextLayer = provider.getOperations().matrixVectorProduct(theta, withBias);
            input = provider.getOperations().function(z_nextLayer, activationFunction);
        }

        return input;

    }

    public void setWeight(int fromLayer, int fromLayerIndex_canIncludeBias, int toNonBiasUnitInNextLayer_zeroBased, double weight) {
        Matrix matrix = thetaMatrices.get(fromLayer);
        matrix.set(toNonBiasUnitInNextLayer_zeroBased, fromLayerIndex_canIncludeBias, weight);
    }

    @Override
    public String toString() {
        StringBuilder bld = new StringBuilder("Neural Net\n");
        for(int i=0; i<thetaMatrices.size(); i++){
            bld.append("Θ ("+i+")\n").append(thetaMatrices.get(i));
        }
        return bld.toString();
    }
}
