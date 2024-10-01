using System.Numerics.Tensors;
using MachineLearning.Model.Layer;
using MachineLearning.Model.Layer.Snapshot;
using MachineLearning.Training.Cost;

namespace MachineLearning.Training.Optimization.Nadam;

public sealed class StringNadamOptimizer : ILayerOptimizer<StringEmbeddingLayer, LayerSnapshots.Embedding>
{
    public StringEmbeddingLayer Layer { get; }
    public ICostFunction CostFunction => Optimizer.CostFunction;
    public NadamOptimizer Optimizer { get; }

    public readonly Matrix GradientCostWeights;
    public readonly Matrix FirstMomentWeights;
    public readonly Matrix SecondMomentWeights;


    public StringNadamOptimizer(NadamOptimizer optimizer, StringEmbeddingLayer layer)
    {
        Optimizer = optimizer;
        Layer = layer;

        GradientCostWeights = Matrix.OfSize(Layer.EmbeddingMatrix);
        FirstMomentWeights = Matrix.OfSize(Layer.EmbeddingMatrix);
        SecondMomentWeights = Matrix.OfSize(Layer.EmbeddingMatrix);
    }

    private readonly Lock _lock = new();
    public void Update(Vector nodeValues, LayerSnapshots.Embedding snapshot)
    {
        throw new NotImplementedException("fix this first (see base Adam)");
        var i = 0;
        lock (_lock)
        {
            foreach (var c in snapshot.LastInput)
            {
                var embedding = GradientCostWeights.RowSpan(Layer.Tokens.IndexOf(c));
                TensorPrimitives.Add(embedding, nodeValues.Slice(i, Layer.EmbeddingSize), embedding);

                i += Layer.EmbeddingSize;
            }
        }
    }

    public void Apply(int dataCounter)
    { 
        var averagedLearningRate = Optimizer.LearningRate / Math.Sqrt(dataCounter);

        (FirstMomentWeights, GradientCostWeights).MapToFirst(FirstMomentEstimate);
        (SecondMomentWeights, GradientCostWeights).MapToFirst(SecondMomentEstimate);
        Layer.EmbeddingMatrix.SubtractToSelf((FirstMomentWeights, SecondMomentWeights, GradientCostWeights).Map(WeightReduction));

        Weight WeightReduction(Weight firstMoment, Weight secondMoment, Weight gradient)
        {
            var mHat = Optimizer.FirstDecayRate * firstMoment / (1 - Math.Pow(Optimizer.FirstDecayRate, Optimizer.Iteration + 1)) + (1 - Optimizer.FirstDecayRate) * gradient / (1 - Math.Pow(Optimizer.FirstDecayRate, Optimizer.Iteration));
            var vHat = secondMoment / (1 - Math.Pow(Optimizer.SecondDecayRate, Optimizer.Iteration));
            return averagedLearningRate * mHat / (Math.Sqrt(vHat) + Optimizer.Epsilon);
        }
        Weight FirstMomentEstimate(Weight lastMoment, Weight gradient)
            => Optimizer.FirstDecayRate * lastMoment + (1 - Optimizer.FirstDecayRate) * gradient;

        Weight SecondMomentEstimate(Weight lastMoment, Weight gradient)
            => Optimizer.SecondDecayRate * lastMoment + (1 - Optimizer.SecondDecayRate) * gradient * gradient;
    }

    public void GradientCostReset()
    {
        GradientCostWeights.ResetZero();
    }

    public void FullReset()
    {
        GradientCostReset();

        FirstMomentWeights.ResetZero();
        SecondMomentWeights.ResetZero();
    }
}
