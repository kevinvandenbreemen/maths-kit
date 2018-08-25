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

        double λ = 1.1; //  TODO    Parameterize!

        if(trainingInputs.size() != expectedOutputs.size()){
            throw new RuntimeException("Number of training inputs not equal to number of training outputs");
        }

        //  Δ matrices for all layers
        List<Matrix> Δs = new ArrayList<>();
        for(int l=0; l<Θ_Matrices.size(); l++){
            Δs.add(provider.matrixOf(Θ_Matrices.get(l).rows(), Θ_Matrices.get(l).cols()-1, 0.0));
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




            //  Compute δ(L-1), δ(L-2), ..., δ(1)
            for(int l = Θ_Matrices.size()-1; l>=0; l--){

                //  First compute g' of activations at activations coming out of layer l
                Vector a_l = activations.get(l+1);
                Vector _1_minus_a_l = provider.getOperations().subtract(
                        provider.vectorOf(1.0, a_l.length()), a_l);
                Vector gPrime_z_l = provider.getOperations().hadamard(a_l, _1_minus_a_l);

                //  Now work out theta(l) transpose * delta of next layer
                Matrix Θ_l = Θ_Matrices.get(l);
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
            for(int l=Θ_Matrices.size()-1; l>=0; l--){
                //  Δ(l) := Δ(l) + δ(l+1)(a(l))ᵀ
                Vector δ_l_plus_1 = δs.get(l+1);
                Matrix deltaL_plus_1_a_l_transpose =
                    provider.getOperations().matrixMatrixProduct(
                            provider.fromVectors(δ_l_plus_1), provider.getOperations().transpose(
                                    provider.fromVectors(activations.get(l))
                            )
                    );

                Δs.set(l, provider.getOperations().add(Δs.get(l), deltaL_plus_1_a_l_transpose));

            }

        }


        //  Compute derivatives
        List<Matrix> finallyTheFuckingDerivatives = new ArrayList<>();
        for(int l=0; l<Δs.size(); l++){
            Matrix D_l = provider.getOperations().function(Δs.get(l), e->e*(1.0/trainingInputs.size()));
            Matrix λΘ_l = provider.getOperations().function(Θ_Matrices.get(l), e->e*λ);

            //  For j != 0, D_l(i,r) = same + lambda(theta_l(i,r))
            for(int j=1; j<D_l.cols(); j++){
                for(int r=0; r<D_l.rows(); r++){
                    D_l.set(r, j, D_l.get(r,j) + λΘ_l.get(r, j));
                }
            }

            finallyTheFuckingDerivatives.add(D_l);
            System.out.println("D("+l+")="+D_l);

            System.out.println("Based on Δ("+l+"), which is "+Δs.get(l));
        }

        //  Now do the goddam cost function
    }

    private double computeCost(List<Vector> trainingInputs, List<Vector> expectedResults, List<Vector> hθ_outputs, double λ){
        double cost = 0.;
        for(int i=0; i<trainingInputs.size(); i++){

            Vector log_hθ_x = provider.getOperations().function(hθ_outputs.get(i), Math::log);
            Vector log_1_minus_hθ_x = provider.getOperations().function(
                        provider.getOperations().subtract(
                                provider.vectorOf(1.0, hθ_outputs.get(i).length()),
                                hθ_outputs.get(i)),
                            Math::log);
            Vector _1_minus_y = provider.getOperations().subtract(
                    provider.vectorOf(1.0, expectedResults.get(i).length()), expectedResults.get(i)
            );

            for(int k=0; k<expectedResults.get(0).length(); k++){
                double innerSum = expectedResults.get(i).entry(k) * log_hθ_x.entry(k) + (_1_minus_y.entry(k) * log_1_minus_hθ_x.entry(k));
            }
        }
        cost *= (-1.0/trainingInputs.size());

        double philmontFactor = 0.0;
        for (int l=0; l<Θ_Matrices.size(); l++){
            Matrix Θ = Θ_Matrices.get(l);
            for(int j=1; j<Θ.cols(); j++){
                for (int r=0; r<Θ.rows(); r++){
                    philmontFactor += Math.pow(Θ.get(r, j), 2);
                }
            }
        }

        philmontFactor *= λ / (2.0*trainingInputs.size());
        return cost + philmontFactor;
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
