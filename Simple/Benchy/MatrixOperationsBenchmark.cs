using System;
using System.Linq;
using System.Numerics;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using SimdVector = System.Numerics.Vector<double>;

namespace Simple.Benchy;

//[SimpleJob]
public class MatrixOperationsBenchmark {
    private double[,] matrix1 = default!;
    private double[,] matrix2 = default!;
    private double[,] result = default!;
    private double[] matrix1Flat = [];
    private double[] matrix2Flat = [];
    private double[] resultFlat = [];
    private Matrix myMatrix1 = default!;
    private Matrix myMatrix2 = default!;
    private Matrix resultMatrix = default!;

    //[Params(16, 128, 512)]
    public int Rows { get; set; } = 128;

    //[Params(16, 128, 512)]
    public int Columns { get; set; } = 128;

    [GlobalSetup]
    public void Setup() {
        var random = new Random();
        matrix1 = new double[Rows, Columns];
        matrix2 = new double[Rows, Columns];
        result = new double[Rows, Columns];
        matrix1Flat = new double[Rows*Columns];
        matrix2Flat = new double[Rows*Columns];
        resultFlat = new double[Rows*Columns];

        for(int i = 0; i < Rows; i++) {
            for(int j = 0; j < Columns; j++) {
                matrix1[i, j] = random.NextDouble();
                matrix2[i, j] = random.NextDouble();
                matrix1Flat[i * Rows + j] = random.NextDouble();
                matrix2Flat[i * Rows + j] = random.NextDouble();
            }
        }

        myMatrix1 = Matrix.Of(Rows, Columns, matrix1Flat);
        myMatrix2 = Matrix.Of(Rows, Columns, matrix2Flat);
        resultMatrix = Matrix.Create(Rows, Columns);
    }

    [Benchmark(Baseline = true)]
    public void Add_Loop() {
        for(int i = 0; i < Rows; i++) {
            for(int j = 0; j < Columns; j++) {
                result[i, j] = matrix1[i, j] + matrix2[i, j];
            }
        }
    }
    
    [Benchmark]
    public void Add_Flat_Loop() {
        for(int i = 0; i < Rows; i++) {
            for(int j = 0; j < Columns; j++) {
                resultFlat[i * Rows + j] = matrix1Flat[i * Rows + j] + matrix2Flat[i * Rows + j];
            }
        }
    }

    [Benchmark]
    public void Add_Flat_SIMD() {
        int vectorSize = SimdVector.Count;

        for(int i = 0; i < Rows; i++) {
            int j = 0;
            for(; j <= Columns - vectorSize; j += vectorSize) {
                var v1 = new SimdVector(matrix1Flat, i * Rows + j);
                var v2 = new SimdVector(matrix2Flat, i * Rows + j);
                var sum = v1 + v2;
                sum.CopyTo(resultFlat, i * Rows + j);
            }
            for(; j < Columns; j++) {
                resultFlat[i * Rows + j] = matrix1Flat[i * Rows + j] + matrix2Flat[i * Rows + j];
            }
        }
    }

    [Benchmark]
    public void Add_Custom() {
        myMatrix1.Add(myMatrix2, resultMatrix);
    }
}
