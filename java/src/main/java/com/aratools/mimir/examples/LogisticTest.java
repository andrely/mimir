package com.aratools.mimir.examples;

import com.aratools.mimir.Binarizer;
import com.aratools.mimir.DataSet;
import no.uib.cipr.matrix.*;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVRecord;

import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.util.List;
import java.util.Random;
import java.util.Vector;
import java.util.stream.IntStream;

public class LogisticTest {
    public static DenseVector onesVector(int size) {
        double vals[] = new double[size];

        for (int i = 0; i < size; i++) {
            vals[i] = 1.0;
        }

        return new DenseVector(vals);
    }

    public static void main(String[] args) {
        List<Double> xVals = new Vector<>();
        List<String> labels = new Vector<>();

        try {
            Reader in = new FileReader("/Users/stinky/Documents/Work/mimir/data/iris.data.csv.txt");
            Iterable<CSVRecord> records = CSVFormat.RFC4180.parse(in);

            for (CSVRecord rec : records) {
                if (rec.size() != 5) {
                    break;
                }

                IntStream.range(0, 4).forEach(i -> {
                    xVals.add(Double.parseDouble(rec.get(i)));
                });

                labels.add(rec.get(4));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        int n = xVals.size() / 4;
        int P = 4;
        double[] vals = new double[xVals.size()];
        IntStream.range(0, xVals.size()).forEach(i -> {
            double col = i % P;
            double row = Math.floor(i / P);
            vals[(int) (col*n + row)] = xVals.get(i);
        });
        Matrix x = new DenseMatrix(n, P, vals, true);

        DataSet<Matrix, List<String>> dataset = new DataSet<>();
        dataset.x = x;
        dataset.y = labels;

        Binarizer<String> bin = new Binarizer<>();
        DenseVector y = Matrices.getColumn(bin.fitTransform(dataset.y), 0);

        Random r = new Random();
        double w0 = r.nextGaussian() * 0.001;
        double wVals[] = new double[4];
        IntStream.range(0, 4).forEach(i -> wVals[i] = r.nextGaussian() * 0.001);
        no.uib.cipr.matrix.Vector w = new DenseVector(wVals, true);

        Matrix xtr = x.transpose(new DenseMatrix(x.numColumns(), x.numRows()));

        for (int i = 0; i < 100; i++) {
            no.uib.cipr.matrix.Vector prob = x.multAdd(w, new DenseVector(n)).add(w0, onesVector(n));
            IntStream.range(0, prob.size()).forEach(j -> prob.set(j, 1.0 / (1.0 + Math.exp(-prob.get(j)))));

            if (i % 10 == 0) {
                double cost = 0.0;

                for (VectorEntry e : prob) {
                    double p = e.get();
                    double val = y.get(e.index());
                    cost -= val*Math.log(p) + (1 - val) * Math.log(1 -p);
                }

                cost = cost / n;

                System.out.println(String.format("cost: %.6f", cost));
            }

            no.uib.cipr.matrix.Vector diff = prob.copy();
            diff.add(-1.0, y);

            double g0 = vectorSum(diff);

            no.uib.cipr.matrix.Vector g = xtr.mult(diff, new DenseVector(P));

            double h0 = 0.0;
            BandMatrix diag = new BandMatrix(n, 0, 0);

            for (VectorEntry e : prob) {
                double val = e.get() * (1.0 - e.get());
                h0 += val;
                diag.set(e.index(), e.index(), val);
            }

            Matrix h = xtr.mult(diag, new DenseMatrix(P, n)).mult(x, new DenseMatrix(P, P));

            w0 -= 0.1*g0/h0;

            no.uib.cipr.matrix.Vector update = h.solve(g, new DenseVector(P));
            update = update.scale(-0.1);

            w = w.add(update);
        }

        System.out.println(String.format("w0: %.4f", w0));
        System.out.println("W:");
        System.out.println(w);
    }

    private static double vectorSum(no.uib.cipr.matrix.Vector v) {
        double total = 0.0;

        for (VectorEntry e : v) {
            total += e.get();
        }

        return total;
    }
}
