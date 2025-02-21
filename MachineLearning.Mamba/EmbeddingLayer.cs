﻿using System.Diagnostics;
using System.Numerics.Tensors;
using MachineLearning.Model.Initialization;
using MachineLearning.Model.Layer;
using MachineLearning.Model.Layer.Initialization;
using MachineLearning.Model.Layer.Snapshot;
using MachineLearning.Training.Cost;
using MachineLearning.Training.Optimization;
using MachineLearning.Training.Optimization.Adam;
using static MachineLearning.Serialization.ModelSerializationHelper;


namespace MachineLearning.Mamba;

public sealed class EmbeddingLayer(int contextSize, Matrix embeddingMatrix) : ILayer
{
    // init randomly with [-0.1; 0.1] or [-0.01; 0.01]
    public Matrix EmbeddingMatrix { get; } = embeddingMatrix;
    // public int OutputNodeCount { get; } = contextSize * embeddingSize;
    public int ContextSize { get; } = contextSize;
    public int EmbeddingSize => EmbeddingMatrix.ColumnCount;
    public int TokenCount => EmbeddingMatrix.RowCount;
    public long ParameterCount => EmbeddingMatrix.FlatCount;

    public EmbeddingLayer(int tokenCount, int contextSize, int embeddingSize)
        : this(contextSize, Matrix.Create(tokenCount, embeddingSize)) { }

    public Matrix Forward(int[] input, Snapshot snapshot)
    {
        // Debug.Assert(input.Length == ContextSize);
        snapshot.Input = input;

        foreach (var i in ..input.Length)
        {
            GetEmbedding(input[i]).CopyTo(snapshot.Output.RowSpan(ContextSize - input.Length + i));
        }

        return snapshot.Output;
    }

    private Span<Weight> GetEmbedding(int index)
    {
        if (index < 0 || index >= EmbeddingMatrix.RowCount)
        {
            throw new ArgumentException($"Unknown token: {index}");
        }

        return EmbeddingMatrix.RowSpan(index);
    }

    public static ErrorState Save(EmbeddingLayer layer, BinaryWriter writer)
    {
        writer.Write(layer.TokenCount);
        writer.Write(layer.ContextSize);
        writer.Write(layer.EmbeddingSize);
        WriteMatrix(layer.EmbeddingMatrix, writer);
        return default;
    }
    public static Result<EmbeddingLayer> Read(BinaryReader reader)
    {
        var tokenCount = reader.ReadInt32();
        var contextSize = reader.ReadInt32();
        var embeddingSize = reader.ReadInt32();
        var matrix = ReadMatrix(tokenCount, embeddingSize, reader);
        return new EmbeddingLayer(contextSize, matrix);
    }

    public ILayerSnapshot CreateSnapshot() => new Snapshot(ContextSize, EmbeddingSize);

    public sealed class Snapshot(int contextSize, int embeddingSize) : ILayerSnapshot
    {
        public int[] Input { get; set; } = [];
        public Matrix Output { get; } = Matrix.Create(contextSize, embeddingSize);
    }

    public sealed class Initializer(Random? random = null) : IInitializer<EmbeddingLayer>
    {
        public Random Random { get; } = random ?? Random.Shared;

        public void Initialize(EmbeddingLayer layer)
        {
            layer.EmbeddingMatrix.MapToSelf(_ => InitializationHelper.RandomInNormalDistribution(Random, 0, 0.05f));
        }
    }
}


public sealed class EmbeddingLayerAdam : ILayerOptimizer<EmbeddingLayer, EmbeddingLayer.Snapshot>
{
    public EmbeddingLayer Layer { get; }
    public ICostFunction CostFunction => Optimizer.CostFunction;
    public AdamOptimizer Optimizer { get; }

    public readonly Matrix Gradient;
    public readonly Matrix FirstMoment;
    public readonly Matrix SecondMoment;
    public readonly Vector GradientCounts;


    public EmbeddingLayerAdam(AdamOptimizer optimizer, EmbeddingLayer layer)
    {
        Optimizer = optimizer;
        Layer = layer;

        Gradient = Matrix.OfSize(Layer.EmbeddingMatrix);
        GradientCounts = Vector.Create(layer.TokenCount);
        FirstMoment = Matrix.OfSize(Layer.EmbeddingMatrix);
        SecondMoment = Matrix.OfSize(Layer.EmbeddingMatrix);
    }

    private readonly Lock _lock = new();
    public void Update(Vector nodeValues, EmbeddingLayer.Snapshot snapshot)
    {
        var outputGradient = Matrix.Of(Layer.ContextSize, Layer.EmbeddingSize, nodeValues);
        lock (_lock)
        {
            foreach (var i in ..snapshot.Input.Length)
            {
                var token = snapshot.Input[i];
                var embeddingGradient = Gradient.RowSpan(token);
                TensorPrimitives.Add(embeddingGradient, outputGradient.RowSpan(i), embeddingGradient);
                GradientCounts[token]++;
            }
        }
    }

    public void Apply(int dataCounter)
    {
        for (int tokenIndex = 0; tokenIndex < Layer.TokenCount; tokenIndex++)
        {
            var count = GradientCounts[tokenIndex];
            if (count > 0)
            {
                var gradientCosts = Gradient.RowRef(tokenIndex);
                gradientCosts.DivideToSelf(count);

                var firstMoment = FirstMoment.RowRef(tokenIndex);
                var secondMoment = SecondMoment.RowRef(tokenIndex);
                (firstMoment, gradientCosts).MapToFirst(FirstMomentEstimate);
                (secondMoment, gradientCosts).MapToFirst(SecondMomentEstimate);
                Layer.EmbeddingMatrix.RowRef(tokenIndex).SubtractToSelf((firstMoment, secondMoment).Map(WeightReduction));
            }
        }
        NumericsDebug.AssertValidNumbers(Gradient);

        Weight WeightReduction(Weight firstMoment, Weight secondMoment)
        {
            var mHat = firstMoment / (1 - MathF.Pow(Optimizer.FirstDecayRate, Optimizer.Iteration));
            var vHat = secondMoment / (1 - MathF.Pow(Optimizer.SecondDecayRate, Optimizer.Iteration));
            return Optimizer.LearningRate * mHat / (MathF.Sqrt(vHat) + Optimizer.Epsilon);
        }

        Weight FirstMomentEstimate(Weight lastMoment, Weight gradient)
            => Optimizer.FirstDecayRate * lastMoment + (1 - Optimizer.FirstDecayRate) * gradient;

        Weight SecondMomentEstimate(Weight lastMoment, Weight gradient)
            => Optimizer.SecondDecayRate * lastMoment + (1 - Optimizer.SecondDecayRate) * gradient * gradient;
    }

    public void GradientCostReset()
    {
        Gradient.ResetZero();
        GradientCounts.ResetZero();
    }

    public void FullReset()
    {
        GradientCostReset();
        FirstMoment.ResetZero();
        SecondMoment.ResetZero();
    }
}