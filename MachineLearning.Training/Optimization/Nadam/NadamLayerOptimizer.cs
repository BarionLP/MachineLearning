using System.Numerics.Tensors;
using MachineLearning.Model.Layer;
using MachineLearning.Model.Layer.Snapshot;
using MachineLearning.Training.Cost;

namespace MachineLearning.Training.Optimization.Nadam;

public sealed class NadamLayerOptimizer : ILayerOptimizer<SimpleLayer>
{
    public SimpleLayer Layer { get; }
    public ICostFunction CostFunction => Optimizer.CostFunction;
    public NadamOptimizer Optimizer { get; }

    public readonly Vector GradientCostBiases;
    public readonly Matrix GradientCostWeights;

    // formula symbol M 
    // exponentially decaying average of past gradients. It is akin to the mean of the gradients.
    public readonly Vector FirstMomentBiases;
    public readonly Matrix FirstMomentWeights;

    // formula symbol V
    // exponentially decaying average of the squared gradients. It is akin to the uncentered variance of the gradients.
    public readonly Vector SecondMomentBiases;
    public readonly Matrix SecondMomentWeights;


    public NadamLayerOptimizer(NadamOptimizer optimizer, SimpleLayer layer)
    {
        Optimizer = optimizer;
        Layer = layer;

        GradientCostBiases = Vector.Create(Layer.OutputNodeCount);
        GradientCostWeights = Matrix.Create(Layer.OutputNodeCount, Layer.InputNodeCount);

        FirstMomentBiases = Vector.Create(Layer.OutputNodeCount);
        FirstMomentWeights = Matrix.Create(Layer.OutputNodeCount, Layer.InputNodeCount);

        SecondMomentBiases = Vector.Create(Layer.OutputNodeCount);
        SecondMomentWeights = Matrix.Create(Layer.OutputNodeCount, Layer.InputNodeCount);
    }

    private readonly Lock _lock = new();
    public void Update(Vector nodeValues, ILayerSnapshot rawSnapshot)
    {
        if (rawSnapshot is not LayerSnapshots.Simple snapshot) throw new UnreachableException();

        // Compute the gradient for weights
        VectorHelper.MultiplyToMatrixTo(nodeValues, snapshot.LastRawInput, snapshot.WeightGradients); // GradientCostWeights.AddInPlaceMultiplied ?

        NumericsDebug.AssertValidNumbers(nodeValues);
        NumericsDebug.AssertValidNumbers(snapshot.WeightGradients);

        lock (_lock)
        {
            GradientCostWeights.AddToSelf(snapshot.WeightGradients);
            GradientCostBiases.AddToSelf(nodeValues);
        }
    }

    public void Apply(int dataCounter)
    {
        // do i need gradient clipping?
        var averagedLearningRate = Optimizer.LearningRate / Math.Sqrt(dataCounter);

        // parallelizing makes no difference
        // Update biases
        (FirstMomentBiases, GradientCostBiases).MapToFirst(FirstMomentEstimate);
        (SecondMomentBiases, GradientCostBiases).MapToFirst(SecondMomentEstimate);
        Layer.Biases.SubtractToSelf((FirstMomentBiases, SecondMomentBiases, GradientCostBiases).Map(WeightReduction));

        // Update weights
        (FirstMomentWeights, GradientCostWeights).MapToFirst(FirstMomentEstimate);
        (SecondMomentWeights, GradientCostWeights).MapToFirst(SecondMomentEstimate);
        Layer.Weights.SubtractToSelf((FirstMomentWeights, SecondMomentWeights, GradientCostWeights).Map(WeightReduction));

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
        GradientCostBiases.ResetZero();
        GradientCostWeights.ResetZero();
    }

    public void FullReset()
    {
        GradientCostBiases.ResetZero();
        FirstMomentBiases.ResetZero();
        SecondMomentBiases.ResetZero();

        GradientCostWeights.ResetZero();
        FirstMomentWeights.ResetZero();
        SecondMomentWeights.ResetZero();
    }
}

public sealed class StringNadamLayerOptimizer : ILayerOptimizer<StringEmbeddingLayer>
{
    public StringEmbeddingLayer Layer { get; }
    public ICostFunction CostFunction => Optimizer.CostFunction;
    public NadamOptimizer Optimizer { get; }

    public readonly Matrix GradientCostWeights;
    public readonly Matrix FirstMomentWeights;
    public readonly Matrix SecondMomentWeights;


    public StringNadamLayerOptimizer(NadamOptimizer optimizer, StringEmbeddingLayer layer)
    {
        Optimizer = optimizer;
        Layer = layer;

        GradientCostWeights = Matrix.OfSize(Layer.EmbeddingMatrix);
        FirstMomentWeights = Matrix.OfSize(Layer.EmbeddingMatrix);
        SecondMomentWeights = Matrix.OfSize(Layer.EmbeddingMatrix);
    }

    private readonly Lock _lock = new();
    public void Update(Vector nodeValues, ILayerSnapshot rawSnapshot)
    {
        if (rawSnapshot is not LayerSnapshots.Embedding snapshot) throw new UnreachableException();

        return; //seems to make things worse

        var i = 0;
        lock (_lock)
        {
            foreach (var c in snapshot.LastInput)
            {
                var embedding = GradientCostWeights.RowSpan(Layer.Tokens.IndexOf(c));
                TensorPrimitives.Add(embedding, nodeValues[i, Layer.EmbeddingSize], embedding);

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
        GradientCostWeights.ResetZero();
        FirstMomentWeights.ResetZero();
        SecondMomentWeights.ResetZero();
    }
}


public sealed class OutputNadamLayerOptimizer : ILayerOptimizer
{
    public ILayer Layer => throw new NotImplementedException();
    public ICostFunction CostFunction => throw new NotImplementedException();

    public void Apply(int dataCounter) { }
    public void FullReset() { }
    public void GradientCostReset() { }
    public void Update(Vector nodeValues, ILayerSnapshot snapshot) => throw new NotImplementedException();
}
