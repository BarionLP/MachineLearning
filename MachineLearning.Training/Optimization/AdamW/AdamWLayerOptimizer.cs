using MachineLearning.Model.Layer;
using MachineLearning.Training.Cost;

namespace MachineLearning.Training.Optimization.AdamW;

public sealed class AdamWLayerOptimizer : ILayerOptimizer
{
    public SimpleLayer Layer { get; }
    public ICostFunction CostFunction => Optimizer.CostFunction;
    public AdamWOptimizer Optimizer { get; }

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


    public AdamWLayerOptimizer(AdamWOptimizer optimizer, SimpleLayer layer)
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
            Console.WriteLine("NaN detected");
        }
        if(snapshot.WeightGradients.AsSpan().Contains(double.NaN))
        {
            Console.WriteLine("NaN detected");
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
        var averagedLearningRate = Optimizer.LearningRate / Math.Sqrt(dataCounter);

        // parallelizing makes no difference
        // Update biases
        (FirstMomentBiases, GradientCostBiases).MapInFirst(FirstMomentEstimate);
        (SecondMomentBiases, GradientCostBiases).MapInFirst(SecondMomentEstimate);
        Layer.Biases.SubtractInPlace((FirstMomentBiases, SecondMomentBiases).Map(WeightReduction));

        // Update weights
        (FirstMomentWeights, GradientCostWeights).MapInFirst(FirstMomentEstimate);
        (SecondMomentWeights, GradientCostWeights).MapInFirst(SecondMomentEstimate);
        var tmp = (FirstMomentWeights, SecondMomentWeights).Map(WeightReduction);
        (Layer.Weights, tmp).MapInFirst(Reduce);


        double Reduce(double original, double reduction)
            => original - reduction - Optimizer.WeightDecayCoefficient * original;
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