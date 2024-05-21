using BenchmarkDotNet.Attributes;
using MachineLearning.Domain.Activation;
using SIMDV = System.Numerics.Vector<double>;
using SIMD = System.Numerics.Vector;
using System.Buffers;
using System.Runtime.CompilerServices;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;


namespace Simple.Benchy;

public class SigmoidBenchmarks {

    IActivationMethod<double> ActivationFunction = SigmoidActivation.Instance;

    double[] InputArray = [];
    Vector<double> InputMathNet = default!;
    SVec InputSVec;

    //[Params(128, 512, 2048)]
    public int size = 512;

    [GlobalSetup]
    public void Setup() {
        InputArray = Enumerable.Range(0, size).Select(n=> Random.Shared.NextDouble()).ToArray();
        InputMathNet = Vector.Build.DenseOfArray(InputArray);
        InputSVec = SVecF.Of(InputArray);
    }


    [Benchmark(Baseline = true)]
    public Vector<double> SigmoidActivation_Current() {
        var activationDerivatives = ActivationFunction.Derivative(InputMathNet);
        return activationDerivatives;
    }

    [Benchmark]
    public double[] SigmoidActivation_Array() {

        var result = new double[InputArray.Length];

        for(int i = 0; i < result.Length; i++) {
            var sigmoid = 1 / (1 + Math.Exp(-InputArray[i]));
            result[i] = sigmoid * (1 - sigmoid);
        }

        return result;
    }
    
    [Benchmark] // (only worth with large arrays)
    public double[] SigmoidActivation_Array_Parallel() {
        var result = new double[InputArray.Length];
        Parallel.For(0, InputArray.Length, (i) =>
        {
            var sigmoid = 1 / (1 + Math.Exp(InputArray[i]));
            result[i] = sigmoid * (1 - sigmoid);
        });

        return result;
    }
    
    // [Benchmark] // same
    // public double[] SigmoidActivation_Array_LINQ() {
    //     return InputArray.Select(v =>
    //     {
    //         var sigmoid = 1 / (1 + Math.Exp(v));
    //         return sigmoid * (1 - sigmoid);
    //     }).ToArray();;
    // }

    [Benchmark] // 0.95
    public void SigmoidActivation_Array_InPlace() {
        for(int i = 0; i < InputArray.Length; i++) {
            var sigmoid = 1 / (1 + Math.Exp(-InputArray[i]));
            InputArray[i] = sigmoid * (1 - sigmoid);
        }
    }


    [Benchmark] //0.93
    public void SigmoidActivation_MathNet_MapInPlace() {
        InputMathNet.MapInplace(v=>{
            var sigmoid = 1 / (1 + Math.Exp(v));
            return sigmoid * (1 - sigmoid);
        });
    }

    // [Benchmark] // 1.12
    // public double[] SigmoidActivation_SIMD() {
    //     var inputLength = InputArray.Length;
    //     var result = new double[inputLength];
    //     var mdSize = SIMDV.Count;
    //     var One = SIMDV.One;

    //     var i = 0;

    //     for(; i <= inputLength - mdSize; i += mdSize) {
    //         var md = new SIMDV(InputArray, i);
    //         var sigmoid = One / (One + (-md).Exp());
    //         var sigmoidDerivative = sigmoid * (One - sigmoid);
    //         sigmoidDerivative.CopyTo(result, i);
    //     }

    //     for(; i < inputLength; i++) {
    //         var sigmoid = 1 / (1 + Math.Exp(-InputArray[i]));
    //         result[i] = sigmoid * (1 - sigmoid);
    //     }

    //     return result;
    // }
    
    SIMDV One = SIMDV.One;
    [Benchmark]
    public void SigmoidActivation_SVec_InPlace() {
        InputSVec.MapInPlace(md => {
            var sigmoid = One / (One + (-md).Exp());
            return sigmoid * (One - sigmoid);
        },
        val => {
            var sigmoid = 1 / (1 + Math.Exp(-val));
            return sigmoid * (1 - sigmoid);
        });
    }
}


public static class SVecF {
    public static SVec Create(int count) {
        return new SVec(count, new double[count]);
    }
    
    public static SVec Of(double[] array) {
        return new SVec(array.Length, array);
    }

    public static void MapInPlace(this SVec left, Func<SIMDV, SIMDV> action, Func<double, double> actionR) {
        int simdCount = SIMDV.Count;
        int i = 0;

        // for(; i <= left.Count - simdCount; i += simdCount) {
        //     action.Invoke(new SIMDV(left[i, simdCount])).CopyTo(left, i);
        // }

        for(; i < left.Count; i++) {
            left[i] = actionR.Invoke(left[i]);
        }
    }

    public static void PointwiseMultiply(this SVec left, SVec right, SVec result) {
        int simdCount = SIMDV.Count;
        int i = 0;

        for(; i <= left.Count - simdCount; i += simdCount) {
            var vec1 = new SIMDV(left[i, simdCount]);
            var vec2 = new SIMDV(right[i, simdCount]);
            (vec1 * vec2).CopyTo(result, i);
        }

        for(; i < left.Count; i++) {
            result[i] = left[i] * right[i];
        }
    }
    
    public static SVec PointwiseMultiply(this SVec left, SVec right) {
        var result = Create(left.Count);
        left.PointwiseMultiply(right, result);
        return result;
    }
    public static void PointwiseMultiplyInPlace(this SVec left, SVec right) {
        left.PointwiseMultiply(right, left);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void CopyTo(this SIMDV vector, SVec destination, int startIndex) {
        if((uint) startIndex >= (uint) destination.Count) {
            throw new ArgumentOutOfRangeException(nameof(startIndex), "start index must be within destination.Count");
        }

        if((destination.Count - startIndex) < SIMDV.Count) {
            throw new ArgumentOutOfRangeException(nameof(destination), "destination is too short");
        }

        Unsafe.WriteUnaligned(ref Unsafe.As<double, byte>(ref destination[startIndex]), vector);
    }

    //[MethodImpl(MethodImplOptions.AggressiveInlining)] // is slower?
    public static SIMDV Exp(this SIMDV vector) {
        Span<double> result = stackalloc double[SIMDV.Count];
        for(int i = 0; i < result.Length; i++) {
            result[i] = Math.Exp(vector[i]);
        }
        return new SIMDV(result);
    }
}

public readonly struct SVec(int count, double[] storage) {
    public int Count { get; } = count;
    private readonly double[] storage = storage;
    public ref double this[int index] => ref storage[index];
    public Span<double> this[int index, int count] => new(storage, index, count); // has built-in bound checks

    public Span<double> AsSpan() => new(storage, 0, Count);
    public static implicit operator Span<double>(SVec vector) => new(vector.storage, 0, vector.Count);
}