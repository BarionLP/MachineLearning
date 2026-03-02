namespace MachineLearning.Training.Optimization.Adam;

public class AdamOptimizer : Optimizer
{
    public static LayerOptimizerRegistry<AdamOptimizer> Registry { get; } = [];
    protected override LayerOptimizerRegistry RegistryGetter => Registry;
    //public required Weight LearningRate { get; init; } = 0.1f;
    public Weight FirstDecayRate { get; init; } = 0.9f;
    public Weight SecondDecayRate { get; init; } = 0.99f; //or 0.999
    public Weight Epsilon { get; init; } = 1e-8f;

    public Weight Iteration { get; set; } = 1; // even when retraining!

    public float FirstMomentEstimate(float lastMoment, float gradient) => FirstDecayRate * lastMoment + (1 - FirstDecayRate) * gradient;
    public float SecondMomentEstimate(float lastMoment, float gradient) => SecondDecayRate * lastMoment + (1 - SecondDecayRate) * gradient * gradient;

    public float WeightReduction(float firstMoment, float secondMoment)
    {
        var mHat = firstMoment / (1 - Weight.Pow(FirstDecayRate, Iteration));
        var vHat = secondMoment / (1 - Weight.Pow(SecondDecayRate, Iteration));
        return LearningRate * mHat / (Weight.Sqrt(vHat) + Epsilon);
    }


    public override void OnBatchCompleted()
    {
        Iteration++;
    }

}