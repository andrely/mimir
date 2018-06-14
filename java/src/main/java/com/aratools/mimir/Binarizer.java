package com.aratools.mimir;

import no.uib.cipr.matrix.DenseMatrix;

import java.util.List;
import java.util.Vector;
import java.util.stream.IntStream;

public class Binarizer<T> {
    private Vector<T> classes;

    public Binarizer fit(List<T> x) {
        classes = new Vector<>();

        for (T e : x) {
            if (!classes.contains(e)) {
                classes.add(e);
            }
        }

        return this;
    }

    public DenseMatrix transform(List<T> x) {
        DenseMatrix m = new DenseMatrix(x.size(), classes.size());

        IntStream.range(0, x.size()).forEach(i -> {
            int pos = classes.indexOf(x.get(i));
            m.set(i, pos, 1.0);
        });

        return m;
    }

    public DenseMatrix fitTransform(List<T> x) {
        fit(x);

        return transform(x);
    }
}
