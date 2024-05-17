using BenchmarkDotNet.Attributes;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using SIMD = System.Numerics.Vector<double>;

namespace Simple.Benchy;

public class ExpBenchmarks {
    double[] array;
    Vector<double> mathnet;
    SVec svec;

    //[Params(128, 512, 2048)]
    public int size = 2048 * 2;

    [GlobalSetup]
    public void Setup() {
        array = Enumerable.Range(0, size).Select(n => Random.Shared.NextDouble()).ToArray();
        mathnet = Vector.Build.DenseOfArray(array);
        svec = SVecF.Of(array);
    }

    [Benchmark(Baseline = true)]
    public void ExpArray() {
        for(int i = 0; i < array.Length; i++) {
            array[i] = Math.Exp(array[i]);
        }
    }
    [Benchmark]
    public void ExpSVecLoop() {
        for(int i = 0; i < svec.Count; i++) {
            array[i] = Math.Exp(svec[i]);
        }
    }
    [Benchmark]
    public void MathNetExp() {
        mathnet.PointwiseExp(mathnet);
    }

    [Benchmark]
    public void ExpSIMDArray() {
        var inputLength = array.Length;
        var vectorWidth = SIMD.Count;
        int i = 0;
        for(; i <= inputLength - vectorWidth; i += vectorWidth) {
            new SIMD(array, i).Exp().CopyTo(array, i);
        }

        for(; i < array.Length; i++) {
            array[i] = Math.Exp(array[i]);
        }
    }

    [Benchmark]
    public void SVecExp() {
        svec.MapInPlace(SVecF.Exp, Math.Exp);
    }
}
