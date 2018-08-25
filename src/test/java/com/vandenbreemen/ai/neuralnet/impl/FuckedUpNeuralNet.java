package com.vandenbreemen.ai.neuralnet.impl;

import com.vandenbreemen.linalg.api.LinalgProvider;
import com.vandenbreemen.linalg.api.Matrix;
import com.vandenbreemen.linalg.api.Vector;
import com.vandenbreemen.linalg.api.VectorFunction;

import java.util.ArrayList;
import java.util.List;

public class FuckedUpNeuralNet {

    private VectorFunction activationFunction;

    private LinalgProvider linalgProvider;

    private List<Matrix> thetaMatrices;

    public FuckedUpNeuralNet(LinalgProvider linalgProvider){
        this.thetaMatrices = new ArrayList<>();
        this.linalgProvider = linalgProvider;
        this.activationFunction = (entry)-> 1.0 / (1.0 + Math.exp(-1.0 * entry));
    }

    public void addLayer(int numInputs, int numOutputs){
        Matrix thetaMatrix = linalgProvider.getMatrix(numOutputs, numInputs+1 /*Accounting for the bias term*/);
        linalgProvider.getOperations().randomEntries(thetaMatrix);
        this.thetaMatrices.add(thetaMatrix);
    }


    public Matrix processInputs(Matrix inputSet) {

        Matrix activationsMatrix = linalgProvider.getOperations().copy(inputSet);
        for(Matrix thetaMatrix : thetaMatrices){

            //  Add the bias term to the inputs
            Matrix tempInputs = linalgProvider.getOperations().transpose(activationsMatrix);
            linalgProvider.getOperations().prependColumn(tempInputs, linalgProvider.vectorOf(1.0, tempInputs.rows()));
            tempInputs = linalgProvider.getOperations().transpose(tempInputs);

            Matrix thetaZ_NextLayer = linalgProvider.getOperations().matrixMatrixProduct(thetaMatrix, tempInputs);
            activationsMatrix = linalgProvider.getOperations().function(thetaZ_NextLayer, activationFunction);
        }

        return activationsMatrix;
    }

    //  TODO    If this does NOT work do NOT give up.  You will need to contrive a series of theta metrices as well as expected outputs and training inputs/outputs and
    //  calculate the cost based on that and verify.
    private double computeCost(double lambda, Matrix trainingInputs, Matrix trainingOutputs, Matrix hThetaXMatrix, List<Matrix> thetaMatrices) {
        double innerSum = 0.0;
        double philmontFactor = 0.0;
        for(int l = 0; l<thetaMatrices.size(); l++){
            for(int i=0; i<thetaMatrices.get(l).cols(); i++){
                for(int j=1; j<thetaMatrices.get(l).rows(); j++){
                    philmontFactor += Math.pow(thetaMatrices.get(l).get(j, i), 2);
                }
            }
        }
        philmontFactor = lambda / (2.0 * trainingInputs.cols());


        for(int i=0; i<trainingInputs.cols(); i++) {

            Vector hTheta_x_i = hThetaXMatrix.columnVector(i);
            Vector log_hTheta_x_i = linalgProvider.getOperations().function(hTheta_x_i, e->Math.log(e));
            Vector log_1_minus_hTheta_x_i = linalgProvider.getOperations().subtract(
                    linalgProvider.vectorOf(1.0, hTheta_x_i.length()), hTheta_x_i
            );
            log_1_minus_hTheta_x_i = linalgProvider.getOperations().function(log_1_minus_hTheta_x_i, e->Math.log(e));

            for (int k = 0; k < trainingOutputs.rows(); k++) {
                double y_k_i = trainingOutputs.get(k, i);

                double sum = y_k_i * log_hTheta_x_i.entry(k);
                sum += (1 - y_k_i) * log_1_minus_hTheta_x_i.entry(k);
                innerSum += sum;
            }
        }

        return (-1.0/trainingInputs.cols()) * innerSum + philmontFactor;
    }

    /**
     * The main event:  Back Propagation!
     * @param trainingInputs
     */
    public void train(Matrix trainingInputs, Matrix trainingOutputs) {
        Matrix activationsMatrix = linalgProvider.getOperations().copy(trainingInputs);

        List<Matrix> bigDELTA_Matrices = new ArrayList<>();
        List<Matrix> activationMatricesForAllLayersBeforeOutput = new ArrayList<>();
        activationMatricesForAllLayersBeforeOutput.add(trainingInputs);
        List<Matrix> dMatrices = new ArrayList<>();

        for(int l=0; l<thetaMatrices.size(); l++){

            Matrix thetaMatrix_at_l = thetaMatrices.get(l);

            //  Add the bias term to the inputs
            Matrix tempInputs = linalgProvider.getOperations().transpose(activationsMatrix);
            linalgProvider.getOperations().prependColumn(tempInputs, linalgProvider.vectorOf(1.0, tempInputs.rows()));
            tempInputs = linalgProvider.getOperations().transpose(tempInputs);

            Matrix thetaZ_NextLayer = linalgProvider.getOperations().matrixMatrixProduct(thetaMatrix_at_l, tempInputs);

            activationsMatrix = linalgProvider.getOperations().function(thetaZ_NextLayer, activationFunction);

            System.out.println("l="+l + " of "+thetaMatrices.size() + " layers");
            if(l <= thetaMatrices.size()-2) {    //  Don't add activation matrix of the output layer!
                activationMatricesForAllLayersBeforeOutput.add(activationsMatrix);
            }
        }

        System.out.println("HIDDEN ACTIVATION MATRICES:");
        activationMatricesForAllLayersBeforeOutput.forEach(matrix -> System.out.println(matrix));
        System.out.println("-----------------------------");

        double lambda = 1.1;
        double cost = computeCost(lambda, trainingInputs, trainingOutputs, activationsMatrix, this.thetaMatrices);
        System.out.println(cost);

        Vector[] yVectors = linalgProvider.toColumnVectors(trainingOutputs);
        Vector[] hThetaVectors = linalgProvider.toColumnVectors(activationsMatrix);

        Vector[] deltaVectors = new Vector[yVectors.length];
        for(int i=0; i<yVectors.length; i++){
            deltaVectors[i] = linalgProvider.getOperations().subtract(hThetaVectors[i], yVectors[i]);
        }

        System.out.println("Expected Outputs:"+trainingOutputs);

        Matrix outputDeltaMatrix = linalgProvider.fromVectors(deltaVectors);
        System.out.println("DELTA:"+outputDeltaMatrix);

        //  Now we compute the preceding layer deltas!
        Matrix delta_l_plus_1 = outputDeltaMatrix;

        for(int l=0; l<thetaMatrices.size(); l++){
            bigDELTA_Matrices.add(linalgProvider.getMatrix(thetaMatrices.get(l).rows(), thetaMatrices.get(l).cols()));
        }

        for (int l = thetaMatrices.size()-1; l>=1; l--){    //  Don't backpropagate to the training set

            System.out.println("l="+l);

            Matrix theta_l__transpose = linalgProvider.getOperations().transpose(thetaMatrices.get(l));
            System.out.println("Theta ("+l+") Transpose:\n"+theta_l__transpose);
            System.out.println("Delta ("+(l+1)+")\n"+delta_l_plus_1);

            Matrix theta_l_transpose_times_delta_l_plus_1 =
                    linalgProvider.getOperations().matrixMatrixProduct(
                            theta_l__transpose, delta_l_plus_1
                    );

            System.out.println("Theta("+l+")T * delta("+(l+1)+"):"+theta_l_transpose_times_delta_l_plus_1);

            Matrix activationsAt_l = activationMatricesForAllLayersBeforeOutput.get(l);
            System.out.println("Activations ("+l+"):"+activationsAt_l);

            activationsAt_l = linalgProvider.getOperations().transpose(activationsAt_l);
            linalgProvider.getOperations().prependColumn(activationsAt_l, linalgProvider.vectorOf(1.0, activationsAt_l.rows()));
            activationsAt_l = linalgProvider.getOperations().transpose(activationsAt_l);
            System.out.println("Activations ("+l+") with bias:"+activationsAt_l);

            Matrix activationDerivatives = linalgProvider.getOperations().hadamard(activationsAt_l,
                        linalgProvider.getOperations().function(activationsAt_l, e->1-e)
                    );
            System.out.println("g' of activations("+l+")"+activationDerivatives);


            Matrix dMatrix = linalgProvider.getOperations().hadamard(theta_l_transpose_times_delta_l_plus_1, activationDerivatives);

            //  Now we've gotta shave off the bias term row
            dMatrix = linalgProvider.getOperations().subMatrixFromRow(dMatrix, 1);

            System.out.println("D(l)\n"+dMatrix);

            delta_l_plus_1 = dMatrix;

        }

        //  Now we need those big delta matrices



    }
}
