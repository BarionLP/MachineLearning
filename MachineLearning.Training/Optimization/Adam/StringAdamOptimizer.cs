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
        var averagedLearningRate = Optimizer.LearningRate / Math.Sqrt(dataCounter);

        for(int i = 0; i < Layer.Tokens.Length; i++)
        {
            var count = GradientCounts[i];
            if(GradientCounts[i] > 1)
            {
                var row = GradientCostWeights.RowRef(i);
                row.DivideToSelf(GradientCounts[i]);
                NumericsDebug.RequireValidNumbers(row);
            }
        }
        NumericsDebug.AssertValidNumbers(GradientCostWeights);

        (FirstMomentWeights, GradientCostWeights).MapToFirst(FirstMomentEstimate);
        (SecondMomentWeights, GradientCostWeights).MapToFirst(SecondMomentEstimate);
        Layer.EmbeddingMatrix.SubtractToSelf((FirstMomentWeights, SecondMomentWeights).Map(WeightReduction));

        double WeightReduction(double firstMoment, double secondMoment)
        {
            var mHat = firstMoment / (1 - Math.Pow(Optimizer.FirstDecayRate, Optimizer.Iteration));
            var vHat = secondMoment / (1 - Math.Pow(Optimizer.SecondDecayRate, Optimizer.Iteration));
            return averagedLearningRate * mHat / (Math.Sqrt(vHat) + Optimizer.Epsilon);
        }
        double FirstMomentEstimate(double lastMoment, double gradient)
            => Optimizer.FirstDecayRate * lastMoment + (1 - Optimizer.FirstDecayRate) * gradient;

        double SecondMomentEstimate(double lastMoment, double gradient)
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
