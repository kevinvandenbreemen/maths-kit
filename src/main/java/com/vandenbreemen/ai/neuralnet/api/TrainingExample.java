package com.vandenbreemen.ai.neuralnet.api;

import com.vandenbreemen.linalg.api.Vector;

public class TrainingExample {

    private Vector input;
    private Vector output;
    private Vector actualOutput;

    public TrainingExample(Vector input, Vector expectedOutput) {
        this.input = input;
        this.output = expectedOutput;
    }

    public Vector getActualOutput() {
        return actualOutput;
    }

    public void setActualOutput(Vector actualOutput) {
        this.actualOutput = actualOutput;
    }

    public Vector getInput() {
        return input;
    }

    public Vector getExpectedOutput() {
        return output;
    }
}
