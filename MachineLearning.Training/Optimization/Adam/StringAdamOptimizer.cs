using System.Numerics.Tensors;
using MachineLearning.Model.Layer;
using MachineLearning.Model.Layer.Snapshot;
using MachineLearning.Training.Cost;

namespace MachineLearning.Training.Optimization.Adam;

public sealed class StringAdamOptimizer : ILayerOptimizer<StringEmbeddingLayer, LayerSnapshots.Embedding>
{
    public StringEmbeddingLayer Layer { get; }
    public ICostFunction CostFunction => Optimizer.CostFunction;
    public AdamOptimizer Optimizer { get; }

    public readonly Matrix GradientCostWeights;
    public readonly Matrix FirstMomentWeights;
    public readonly Matrix SecondMomentWeights;
    public readonly Vector GradientCounts;


    public StringAdamOptimizer(AdamOptimizer optimizer, StringEmbeddingLayer layer)
    {
        Optimizer = optimizer;
        Layer = layer;

        GradientCostWeights = Matrix.OfSize(Layer.EmbeddingMatrix);
        GradientCounts = Vector.Create(layer.Tokens.Length);
        FirstMomentWeights = Matrix.OfSize(Layer.EmbeddingMatrix);
        SecondMomentWeights = Matrix.OfSize(Layer.EmbeddingMatrix);
    }

    private readonly Lock _lock = new();
    public void Update(Vector nodeValues, LayerSnapshots.Embedding snapshot)
    {
        var i = 0;
        lock (_lock)
        {
            foreach (var c in snapshot.LastInput)
            {
                var index = Layer.Tokens.IndexOf(c);
                var embedding = GradientCostWeights.RowSpan(index);
                TensorPrimitives.Add(embedding, nodeValues.Slice(i, Layer.EmbeddingSize), embedding);
                GradientCounts[index]++;

                i += Layer.EmbeddingSize;
            }
        }
    }

    public void Apply(int dataCounter)
    { 
        var averagedLearningRate = Optimizer.LearningRate / MathF.Sqrt(dataCounter);

        for(int tokenIndex = 0; tokenIndex < Layer.Tokens.Length; tokenIndex++)
        {
            var count = GradientCounts[tokenIndex];
            if(count > 0)
            {
                var gradientCosts = GradientCostWeights.RowRef(tokenIndex);
                gradientCosts.DivideToSelf(count);

                var firstMoment = FirstMomentWeights.RowRef(tokenIndex);
                var secondMoment = SecondMomentWeights.RowRef(tokenIndex);
                (firstMoment, gradientCosts).MapToFirst(FirstMomentEstimate);
                (secondMoment, gradientCosts).MapToFirst(SecondMomentEstimate);
                Layer.EmbeddingMatrix.RowRef(tokenIndex).SubtractToSelf((firstMoment, secondMoment).Map(WeightReduction));
            }
        }
        NumericsDebug.AssertValidNumbers(GradientCostWeights);

        Weight WeightReduction(Weight firstMoment, Weight secondMoment)
        {
            var mHat = firstMoment / (1 - MathF.Pow(Optimizer.FirstDecayRate, Optimizer.Iteration));
            var vHat = secondMoment / (1 - MathF.Pow(Optimizer.SecondDecayRate, Optimizer.Iteration));
            return averagedLearningRate * mHat / (MathF.Sqrt(vHat) + Optimizer.Epsilon);
        }

        Weight FirstMomentEstimate(Weight lastMoment, Weight gradient)
            => Optimizer.FirstDecayRate * lastMoment + (1 - Optimizer.FirstDecayRate) * gradient;

        Weight SecondMomentEstimate(Weight lastMoment, Weight gradient)
            => Optimizer.SecondDecayRate * lastMoment + (1 - Optimizer.SecondDecayRate) * gradient * gradient;
    }

    public void GradientCostReset()
    {
        GradientCostWeights.ResetZero();
        GradientCounts.ResetZero();
    }

    public void FullReset()
    {
        GradientCostReset();
        FirstMomentWeights.ResetZero();
        SecondMomentWeights.ResetZero();
    }
}
