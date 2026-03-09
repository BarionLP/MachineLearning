namespace ML.Core.Training;

public sealed class AdamOptimizer : Optimizer
{
    public static ModuleOptimizerRegistry<AdamOptimizer> Registry { get; } = [];
    protected override ModuleOptimizerRegistry RegistryGetter => Registry;
    public Weight FirstDecayRate { get; init; } = 0.9f;
    public Weight SecondDecayRate { get; init; } = 0.99f; //or 0.999
    public Weight Epsilon { get; init; } = 1e-8f;

    public Weight Iteration
    {
        get;
        set
        {
            field = value;
            CurrentFirstCorrection = 1 - Weight.Pow(FirstDecayRate, Iteration);
            CurrentSecondCorrection = 1 - Weight.Pow(SecondDecayRate, Iteration);
        }
    }

    public Weight CurrentFirstCorrection { get; private set; }
    public Weight CurrentSecondCorrection { get; private set; }

    public Weight FirstMomentEstimate(Weight lastMoment, Weight gradient) => FirstDecayRate * lastMoment + (1 - FirstDecayRate) * gradient;
    public Weight SecondMomentEstimate(Weight lastMoment, Weight gradient) => SecondDecayRate * lastMoment + (1 - SecondDecayRate) * gradient * gradient;
    public SimdVector FirstMomentEstimate(SimdVector lastMoment, SimdVector gradient) => FirstDecayRate * lastMoment + (1 - FirstDecayRate) * gradient;
    public SimdVector SecondMomentEstimate(SimdVector lastMoment, SimdVector gradient) => SecondDecayRate * lastMoment + (1 - SecondDecayRate) * gradient * gradient;

    public Weight WeightReduction(Weight firstMoment, Weight secondMoment)
    {
        var mHat = firstMoment / CurrentFirstCorrection;
        var vHat = secondMoment / CurrentSecondCorrection;
        return LearningRate * mHat / (Weight.Sqrt(vHat) + Epsilon);
    }
    public SimdVector WeightReduction(SimdVector firstMoment, SimdVector secondMoment)
    {
        var mHat = firstMoment / CurrentFirstCorrection;
        var vHat = secondMoment / CurrentSecondCorrection;
        return LearningRate * mHat / (SimdVectorHelper.SquareRoot(vHat) + SimdVectorHelper.Create(Epsilon));
    }

    public override void Init()
    {
        Iteration = 1;  // even when retraining!
    }

    public override void OnBatchCompleted()
    {
        Iteration++;
    }
}
