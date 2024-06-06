using MachineLearning.Model.Layer;
using MachineLearning.Training.Cost;

namespace MachineLearning.Training.Optimization.Layer;

public sealed class AdamLayerOptimizer : ILayerOptimizer
{
    public SimpleLayer Layer { get; }
    public ICostFunction CostFunction => Optimizer.Config.CostFunction;
    public AdamOptimizer Optimizer { get; }

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


    public AdamLayerOptimizer(AdamOptimizer optimizer, SimpleLayer layer)
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

    private readonly object _lock = new();
    public void Update(Vector nodeValues, LayerSnapshot snapshot)
    {
        // Compute the gradient for weights
        VectorHelper.MultiplyToMatrix(nodeValues, snapshot.LastRawInput, snapshot.WeightGradients); // GradientCostWeights.AddInPlaceMultiplied ?
#if DEBUG
        if(nodeValues.AsSpan().Contains(double.NaN))
        {
            Console.WriteLine();
        }
        if(snapshot.WeightGradients.AsSpan().Contains(double.NaN))
        {
            Console.WriteLine();
        }
#endif

        lock(_lock)
        { 
            GradientCostWeights.AddInPlace(snapshot.WeightGradients);
            GradientCostBiases.AddInPlace(nodeValues);
        }
    }

    public void Apply(int dataCounter)
    {
        // do i need gradient clipping?
        var averagedLearningRate = Optimizer.Config.LearningRate / Math.Sqrt(dataCounter);

        // parallelizing makes no difference
        // Update biases
        (FirstMomentBiases, GradientCostBiases).MapInFirst(FirstMomentEstimate);
        (SecondMomentBiases, GradientCostBiases).MapInFirst(SecondMomentEstimate);
        Layer.Biases.SubtractInPlace((FirstMomentBiases, SecondMomentBiases).Map(WeightReduction));

        // Update weights
        (FirstMomentWeights, GradientCostWeights).MapInPlaceOnFirst(FirstMomentEstimate);
        (SecondMomentWeights, GradientCostWeights).MapInPlaceOnFirst(SecondMomentEstimate);
        Layer.Weights.SubtractInPlace((FirstMomentWeights, SecondMomentWeights).Map(WeightReduction));

        double WeightReduction(double firstMoment, double secondMoment)
        {
            var mHat = firstMoment / (1 - Math.Pow(Optimizer.Config.FirstDecayRate, Optimizer.Iteration));
            var vHat = secondMoment / (1 - Math.Pow(Optimizer.Config.SecondDecayRate, Optimizer.Iteration));
            return averagedLearningRate * mHat / (Math.Sqrt(vHat) + Optimizer.Config.Epsilon);
        }
        double FirstMomentEstimate(double lastMoment, double gradient) 
            => (Optimizer.Config.FirstDecayRate * lastMoment) + (1 - Optimizer.Config.FirstDecayRate) * gradient;

        double SecondMomentEstimate(double lastMoment, double gradient) 
            => (Optimizer.Config.SecondDecayRate * lastMoment) + (1 - Optimizer.Config.SecondDecayRate) * gradient * gradient;
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