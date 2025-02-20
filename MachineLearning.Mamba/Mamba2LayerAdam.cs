using MachineLearning.Training.Cost;
using MachineLearning.Training.Optimization;
using MachineLearning.Training.Optimization.Adam;

namespace MachineLearning.Mamba;

public sealed class Mamba2LayerAdam : ILayerOptimizer<Mamba2Layer, Mamba2Layer.Snapshot>
{
    public Mamba2Layer Layer { get; }
    public ICostFunction CostFunction => Optimizer.CostFunction;
    public AdamOptimizer Optimizer { get; }

    public Vector GradientAlpha { get; }
    public Matrix GradientB { get; }
    public Matrix GradientC { get; }

    // formula symbol M 
    // exponentially decaying average of past gradients. It is akin to the mean of the gradients.
    public readonly Vector FirstMomentAlpha;
    public readonly Matrix FirstMomentB;
    public readonly Matrix FirstMomentC;

    // formula symbol V
    // exponentially decaying average of the squared gradients. It is akin to the uncentered variance of the gradients.
    public readonly Vector SecondMomentAlpha;
    public readonly Matrix SecondMomentB;
    public readonly Matrix SecondMomentC;


    public Mamba2LayerAdam(AdamOptimizer optimizer, Mamba2Layer layer)
    {
        Optimizer = optimizer;
        Layer = layer;

        GradientAlpha = Vector.Create(Layer.SequenceLength);
        GradientB = Matrix.Create(Layer.SequenceLength, Layer.StateDimensions);
        GradientC = Matrix.Create(Layer.SequenceLength, Layer.StateDimensions);

        FirstMomentAlpha = Vector.Create(Layer.SequenceLength);
        FirstMomentB = Matrix.Create(Layer.SequenceLength, Layer.StateDimensions);
        FirstMomentC = Matrix.Create(Layer.SequenceLength, Layer.StateDimensions);

        SecondMomentAlpha = Vector.Create(Layer.SequenceLength);
        SecondMomentB = Matrix.Create(Layer.SequenceLength, Layer.StateDimensions);
        SecondMomentC = Matrix.Create(Layer.SequenceLength, Layer.StateDimensions);
    }

    private readonly Lock _lock = new();
    public void Update(Vector nodeValues, Mamba2Layer.Snapshot snapshot)
    {
        Layer.BackwardPass(snapshot, nodeValues);
        // Compute the gradient for weights
        //VectorHelper.MultiplyToMatrixTo(nodeValues, snapshot.LastRawInput, snapshot.WeightGradients); // GradientCostWeights.AddInPlaceMultiplied ?

        NumericsDebug.AssertValidNumbers(nodeValues);
        NumericsDebug.AssertValidNumbers(snapshot.GradientAlpha);
        NumericsDebug.AssertValidNumbers(snapshot.GradientB);
        NumericsDebug.AssertValidNumbers(snapshot.GradientC);

        lock (_lock)
        {
            GradientAlpha.AddToSelf(snapshot.GradientAlpha);
            GradientB.AddToSelf(snapshot.GradientB);
            GradientC.AddToSelf(snapshot.GradientC);
        }
    }

    // update child methods
    public void Apply(int dataCounter)
    {
        // do i need gradient clipping?
        var averagedLearningRate = Optimizer.LearningRate;

        // Update biases
        (FirstMomentAlpha, GradientAlpha).MapToFirst(FirstMomentEstimate);
        (SecondMomentAlpha, GradientAlpha).MapToFirst(SecondMomentEstimate);
        Layer.Alpha.SubtractToSelf((FirstMomentAlpha, SecondMomentAlpha).Map(WeightReduction));

        (FirstMomentB, GradientB).MapToFirst(FirstMomentEstimate);
        (SecondMomentB, GradientB).MapToFirst(SecondMomentEstimate);
        Layer.B.SubtractToSelf((FirstMomentB, SecondMomentB).Map(WeightReduction));

        (FirstMomentC, GradientC).MapToFirst(FirstMomentEstimate);
        (SecondMomentC, GradientC).MapToFirst(SecondMomentEstimate);
        Layer.C.SubtractToSelf((FirstMomentC, SecondMomentC).Map(WeightReduction));

        Weight WeightReduction(Weight firstMoment, Weight secondMoment)
        {
            var mHat = firstMoment / (1 - Weight.Pow(Optimizer.FirstDecayRate, Optimizer.Iteration));
            var vHat = secondMoment / (1 - Weight.Pow(Optimizer.SecondDecayRate, Optimizer.Iteration));
            return averagedLearningRate * mHat / (Weight.Sqrt(vHat) + Optimizer.Epsilon);
        }
        Weight FirstMomentEstimate(Weight lastMoment, Weight gradient)
            => Optimizer.FirstDecayRate * lastMoment + (1 - Optimizer.FirstDecayRate) * gradient;

        Weight SecondMomentEstimate(Weight lastMoment, Weight gradient)
            => Optimizer.SecondDecayRate * lastMoment + (1 - Optimizer.SecondDecayRate) * gradient * gradient;
    }

    public void GradientCostReset()
    {
        GradientAlpha.ResetZero();
        GradientB.ResetZero();
        GradientC.ResetZero();
    }

    public void FullReset()
    {
        GradientCostReset();

        FirstMomentAlpha.ResetZero();
        SecondMomentAlpha.ResetZero();

        FirstMomentB.ResetZero();
        SecondMomentB.ResetZero();
        FirstMomentC.ResetZero();
        SecondMomentC.ResetZero();
    }
}