using MachineLearning.Model.Layer;
using MachineLearning.Training.Optimization.Adam;

namespace MachineLearning.Training.Optimization.AdamW;

public sealed class SimpleAdamWOptimizer(AdamWOptimizer optimizer, SimpleLayer layer) : SimpleAdamOptimizer(optimizer, layer)
{
    public new AdamWOptimizer Optimizer { get; } = optimizer;

    public override void Apply(int dataCounter)
    {
        var averagedLearningRate = Optimizer.LearningRate / MathF.Sqrt(dataCounter);

        (FirstMomentBiases, GradientCostBiases).MapToFirst(FirstMomentEstimate);
        (SecondMomentBiases, GradientCostBiases).MapToFirst(SecondMomentEstimate);
        Layer.Biases.SubtractToSelf((FirstMomentBiases, SecondMomentBiases).Map(WeightReduction));

        (FirstMomentWeights, GradientCostWeights).MapToFirst(FirstMomentEstimate);
        (SecondMomentWeights, GradientCostWeights).MapToFirst(SecondMomentEstimate);
        var tmp = (FirstMomentWeights, SecondMomentWeights).Map(WeightReduction);
        (Layer.Weights, tmp).MapToFirst(Reduce);

        Weight Reduce(Weight original, Weight reduction)
            => original - reduction - Optimizer.WeightDecayCoefficient * original;

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
}