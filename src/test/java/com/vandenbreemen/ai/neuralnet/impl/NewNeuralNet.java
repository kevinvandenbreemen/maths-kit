package com.vandenbreemen.ai.neuralnet.impl;

import com.vandenbreemen.linalg.api.LinalgProvider;
import com.vandenbreemen.linalg.api.Matrix;
import com.vandenbreemen.linalg.api.Vector;
import com.vandenbreemen.linalg.api.VectorFunction;

import java.util.ArrayList;
import java.util.List;

public class NewNeuralNet {

    private VectorFunction activationFunction;

    private List<Matrix> Θ_Matrices;

    private LinalgProvider provider;

    public NewNeuralNet(LinalgProvider provider) {
        this.provider = provider;
        this.Θ_Matrices = new ArrayList<>();
        this.activationFunction = (entry)-> 1.0 / (1.0 + Math.exp(-1.0 * entry));
    }

    public void addLayer(int numInputs, int numOutputs){
        Matrix matrix = provider.getMatrix(numOutputs, numInputs+1);    //  Add 1 to account for the bias unit
        matrix = provider.getOperations().randomEntries(matrix);
        Θ_Matrices.add(matrix);
    }

    /**
     *
     * @param x
     */
    public Vector compute_hTheta(Vector x){

        Vector input = x;

        for(Matrix Θ : Θ_Matrices){
            input = computeActivations(Θ, input);
        }

        return input;

    }

    private Vector computeActivations(Matrix theta, Vector input) {
        Vector withBias = provider.getOperations().prependEntry(input, 1.0);
        Vector z_nextLayer = provider.getOperations().matrixVectorProduct(theta, withBias);
        input = provider.getOperations().function(z_nextLayer, activationFunction);
        return input;
    }

    public void train(List<Vector> trainingInputs, List<Vector> expectedOutputs){
        if(trainingInputs.size() != expectedOutputs.size()){
            throw new RuntimeException("Number of training inputs not equal to number of training outputs");
        }


        for(int i = 0; i<trainingInputs.size(); i++){   //  "for i=1 to m"

            //  Activations (0-based)
            List<Vector> activations = new ArrayList<>();

            //  Set a(1) = x(i)
            activations.add(trainingInputs.get(i));
            Vector activationVector = trainingInputs.get(i);

            //  Perform forward propagation to compute a(l) for l = 2,3,...L
            for(Matrix Θ : Θ_Matrices){
                activationVector = computeActivations(Θ, activationVector);
                activations.add(activationVector);
            }

            //  Using y(i), compute δ(L) = a(L) - y(i)
            List<Vector> δs = new ArrayList<>();
            Vector δ_L = provider.getOperations().subtract(activations.get(activations.size()-1), expectedOutputs.get(i));
            δs.add(0, δ_L);

            //  Δ matrices for all layers
            List<Matrix> Δs = new ArrayList<>();
            for(int l=0; l<Θ_Matrices.size(); l++){
                Δs.add(provider.matrixOf(Θ_Matrices.get(l).rows(), Θ_Matrices.get(l).cols()-1, 0.0));
            }


            //  Compute δ(L-1), δ(L-2), ..., δ(1)
            for(int l = Θ_Matrices.size()-1; l>=0; l--){

                //  First compute g' of activations at activations coming out of layer l
                Vector a_l = activations.get(l+1);
                Vector _1_minus_a_l = provider.getOperations().subtract(
                        provider.vectorOf(1.0, a_l.length()), a_l);
                Vector gPrime_z_l = provider.getOperations().hadamard(a_l, _1_minus_a_l);

                //  Now work out theta(l) transpose * delta of next layer
                Matrix Θ_l = Θ_Matrices.get(l);
                System.out.println("Θ("+l+")="+Θ_l);
                System.out.println("δ("+(l+1)+")="+δs.get(0));
                Vector δ = provider.getOperations().matrixVectorProduct(
                        provider.getOperations().subMatrixFromRow(
                                provider.getOperations().transpose(Θ_l),
                                1
                        )
                            ,
                        δs.get(0)
                );
                δs.add(0, δ);

            }

            //  Compute the Δs
            for(int l=0; l<Θ_Matrices.size(); l++){
                //  Δ(l) := Δ(l) + δ(l+1)(a(l))ᵀ
                Vector δ_l_plus_1 = δs.get(l+1);
                System.out.println("δ("+(l+1)+")="+δ_l_plus_1);
                System.out.println("a("+l+")="+activations.get(l));
                Matrix deltaL_plus_1_a_l_transpose =
                    provider.getOperations().matrixMatrixProduct(
                            provider.fromVectors(δ_l_plus_1), provider.getOperations().transpose(
                                    provider.fromVectors(activations.get(l))
                            )
                    );
                System.out.println("δ(l+1)(a(l))ᵀ"+deltaL_plus_1_a_l_transpose);
                System.out.println("Δ("+l+")="+Δs.get(l));

                Δs.set(l, provider.getOperations().add(Δs.get(l), deltaL_plus_1_a_l_transpose));

                System.out.println("Δ("+l+")="+Δs.get(l));
            }

            System.out.println("TOTAL Δ="+Δs.size());
            System.out.println(Δs);

            //  Compute derivatives


        }
    }



    public void setWeight(int fromLayer, int fromLayerIndex_canIncludeBias, int toNonBiasUnitInNextLayer_zeroBased, double weight) {
        Matrix matrix = Θ_Matrices.get(fromLayer);
        matrix.set(toNonBiasUnitInNextLayer_zeroBased, fromLayerIndex_canIncludeBias, weight);
    }

    @Override
    public String toString() {
        StringBuilder bld = new StringBuilder("Neural Net\n");
        for(int i = 0; i< Θ_Matrices.size(); i++){
            bld.append("Θ ("+i+")\n").append(Θ_Matrices.get(i));
        }
        return bld.toString();
    }
}
