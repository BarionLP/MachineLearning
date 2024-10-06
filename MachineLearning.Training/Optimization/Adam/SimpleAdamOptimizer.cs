using MachineLearning.Model.Layer;
using MachineLearning.Model.Layer.Snapshot;
using MachineLearning.Training.Cost;

namespace MachineLearning.Training.Optimization.Adam;

public class SimpleAdamOptimizer : ILayerOptimizer<SimpleLayer, LayerSnapshots.Simple>
{
    public SimpleLayer Layer { get; }
    public ICostFunction CostFunction => Optimizer.CostFunction;
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


    public SimpleAdamOptimizer(AdamOptimizer optimizer, SimpleLayer layer)
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
    public void Update(Vector nodeValues, LayerSnapshots.Simple snapshot)
    {
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

    // update child methods
    public virtual void Apply(int dataCounter)
    {
        // do i need gradient clipping?
        var averagedLearningRate = Optimizer.LearningRate / Math.Sqrt(dataCounter);

        // parallelizing makes no difference
        // Update biases
        (FirstMomentBiases, GradientCostBiases).MapToFirst(FirstMomentEstimate);
        (SecondMomentBiases, GradientCostBiases).MapToFirst(SecondMomentEstimate);
        Layer.Biases.SubtractToSelf((FirstMomentBiases, SecondMomentBiases).Map(WeightReduction));

        // Update weights
        (FirstMomentWeights, GradientCostWeights).MapToFirst(FirstMomentEstimate);
        (SecondMomentWeights, GradientCostWeights).MapToFirst(SecondMomentEstimate);
        Layer.Weights.SubtractToSelf((FirstMomentWeights, SecondMomentWeights).Map(WeightReduction));

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
