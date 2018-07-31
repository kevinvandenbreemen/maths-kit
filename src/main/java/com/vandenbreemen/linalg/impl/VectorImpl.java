package com.vandenbreemen.linalg.impl;

import com.vandenbreemen.linalg.api.Vector;

public class VectorImpl implements Vector {

    private double[] entries;

    public VectorImpl(double[] entries) {
        this.entries = new double[entries.length];
        System.arraycopy(entries, 0, this.entries, 0, entries.length);
    }

    @Override
    public int length() {
        return this.entries.length;
    }

    @Override
    public double entry(int position) {
        return this.entries[position];
    }
}
