using MachineLearning.Model.Layer;
using MachineLearning.Training.Optimization.Adam;

namespace MachineLearning.Training.Optimization.AdamW;

public sealed class SimpleAdamWOptimizer(AdamWOptimizer optimizer, SimpleLayer layer) : SimpleAdamOptimizer(optimizer, layer)
{
    public new AdamWOptimizer Optimizer { get; } = optimizer;

    public override void Apply(int dataCounter)
    {
        var averagedLearningRate = Optimizer.LearningRate / Math.Sqrt(dataCounter);

        (FirstMomentBiases, GradientCostBiases).MapToFirst(FirstMomentEstimate);
        (SecondMomentBiases, GradientCostBiases).MapToFirst(SecondMomentEstimate);
        Layer.Biases.SubtractToSelf((FirstMomentBiases, SecondMomentBiases).Map(WeightReduction));

        (FirstMomentWeights, GradientCostWeights).MapToFirst(FirstMomentEstimate);
        (SecondMomentWeights, GradientCostWeights).MapToFirst(SecondMomentEstimate);
        var tmp = (FirstMomentWeights, SecondMomentWeights).Map(WeightReduction);
        (Layer.Weights, tmp).MapToFirst(Reduce);

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
}