using System;
using System.Linq;
using System.Numerics;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using Vector = System.Numerics.Vector<double>;
//using MathNetVector = MathNet.Numerics.LinearAlgebra.Vector<double>;

namespace Simple.Benchy;

[SimpleJob]
public class MatrixOperationsBenchmark {
    private double[,] matrix1 = default!;
    private double[,] matrix2 = default!;
    private double[] matrix1Flat = [];
    private double[] matrix2Flat = [];
    private Matrix<double> mathNetMatrix1 = default!;
    private Matrix<double> mathNetMatrix2 = default!;

    [Params(16, 128, 512)]
    public int Rows { get; set; }
    
    [Params(16, 128, 512)]
    public int Columns { get; set; }

    [GlobalSetup]
    public void Setup() {
        var random = new Random();
        matrix1 = new double[Rows, Columns];
        matrix2 = new double[Rows, Columns];
        matrix1Flat = new double[Rows*Columns];
        matrix2Flat = new double[Rows*Columns];
        for(int i = 0; i < Rows; i++) {
            for(int j = 0; j < Columns; j++) {
                matrix1[i, j] = random.NextDouble();
                matrix2[i, j] = random.NextDouble();
                matrix1Flat[i * Rows + j] = random.NextDouble();
                matrix2Flat[i * Rows + j] = random.NextDouble();
            }
        }

        mathNetMatrix1 = DenseMatrix.OfArray(matrix1);
        mathNetMatrix2 = DenseMatrix.OfArray(matrix2);
    }

    [Benchmark(Baseline = true)]
    public double[,] AddMatrices_SimpleLoop() {
        var result = new double[Rows, Columns];
        for(int i = 0; i < Rows; i++) {
            for(int j = 0; j < Columns; j++) {
                result[i, j] = matrix1[i, j] + matrix2[i, j];
            }
        }
        return result;
    }
    
    [Benchmark]
    public double[] AddFlatMatrices_SimpleLoop() {
        var result = new double[Rows * Columns];
        for(int i = 0; i < Rows; i++) {
            for(int j = 0; j < Columns; j++) {
                result[i * Rows + j] = matrix1Flat[i * Rows + j] + matrix2Flat[i * Rows + j];
            }
        }
        return result;
    }

    [Benchmark]
    public double[] AddMatricesFlat_SIMD() {
        var result = new double[Rows * Columns];
        int vectorSize = Vector.Count;

        for(int i = 0; i < Rows; i++) {
            int j = 0;
            for(; j <= Columns - vectorSize; j += vectorSize) {
                var v1 = new Vector(matrix1Flat, i * Rows + j);
                var v2 = new Vector(matrix2Flat, i * Rows + j);
                var sum = v1 + v2;
                sum.CopyTo(result, i * Rows + j);
            }
            for(; j < Columns; j++) {
                result[i * Rows + j] = matrix1Flat[i * Rows + j] + matrix2Flat[i * Rows + j];
            }
        }
        return result;
    }

    [Benchmark]
    public Matrix<double> AddMatrices_MathNet() {
        return mathNetMatrix1 + mathNetMatrix2;
    }

    [Benchmark]
    public double[,] MultiplyMatrices_SimpleLoop() {
        var result = new double[Rows, Columns];
        for(int i = 0; i < Rows; i++) {
            for(int j = 0; j < Columns; j++) {
                result[i, j] = matrix1[i, j] * matrix2[i, j];
            }
        }
        return result;
    }

    [Benchmark]
    public double[] MultiplyMatricesFlat_SIMD() {
        var result = new double[Rows * Columns];
        int vectorSize = Vector.Count;

        for(int i = 0; i < Rows; i++) {
            int j = 0;
            for(; j <= Columns - vectorSize; j += vectorSize) {
                var v1 = new Vector(matrix1Flat, i * Rows + j);
                var v2 = new Vector(matrix2Flat, i * Rows + j);
                var product = v1 * v2;
                product.CopyTo(result, i * Rows + j);
            }
            for(; j < Columns; j++) {
                result[i * Rows + j] = matrix1Flat[i * Rows + j] * matrix2Flat[i * Rows + j];
            }
        }
        return result;
    }

    [Benchmark]
    public Matrix<double> MultiplyMatrices_MathNet() {
        return mathNetMatrix1.PointwiseMultiply(mathNetMatrix2);
    }
}
