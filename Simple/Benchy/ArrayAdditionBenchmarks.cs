using System;
using System.Linq;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using System.Numerics;
using Vector = System.Numerics.Vector<double>;
using MathNetVector = MathNet.Numerics.LinearAlgebra.Vector<double>;

namespace Simple.Benchy;

public class ArrayAdditionBenchmarks {
    private double[] array1 = [];
    private double[] array2 = [];

    private MathNetVector vec1 = default!;
    private MathNetVector vec2 = default!;

    [GlobalSetup]
    public void Setup() {
        int size = 1000;
        array1 = new double[size];
        array2 = new double[size];
        for(int i = 0; i < size; i++) {
            array1[i] = i;
            array2[i] = size - i;
        }
        vec1 = MathNetVector.Build.DenseOfArray(array1);
        vec2 = MathNetVector.Build.DenseOfArray(array2);
    }

    [Benchmark(Baseline = true)]
    public double[] SimpleLoopAddition() {
        double[] result = new double[array1.Length];
        for(int i = 0; i < array1.Length; i++) {
            result[i] = array1[i] + array2[i];
        }
        return result;
    }

    [Benchmark]
    public double[] SimdAddition() {

        double[] result = new double[array1.Length];
        int simdLength = Vector.Count;
        int i = 0;

        for(; i <= array1.Length - simdLength; i += simdLength) {
            var vec1 = new Vector(array1, i);
            var vec2 = new Vector(array2, i);
            (vec1 + vec2).CopyTo(result, i);
        }

        for(; i < array1.Length; i++) {
            result[i] = array1[i] + array2[i];
        }

        return result;
    }
   
    
    [Benchmark]
    public MathNetVector MathNetAddition() {
        return vec1 + vec2;
    }
}
