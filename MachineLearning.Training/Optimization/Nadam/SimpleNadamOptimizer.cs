using MachineLearning.Model.Layer;
using MachineLearning.Training.Optimization.Adam;

namespace MachineLearning.Training.Optimization.Nadam;

public sealed class SimpleNadamOptimizer(NadamOptimizer optimizer, SimpleLayer layer) : SimpleAdamOptimizer(optimizer, layer)
{
    public override void Apply(int dataCounter)
    {
        var averagedLearningRate = Optimizer.LearningRate / Math.Sqrt(dataCounter);

        (FirstMomentBiases, GradientCostBiases).MapToFirst(FirstMomentEstimate);
        (SecondMomentBiases, GradientCostBiases).MapToFirst(SecondMomentEstimate);
        Layer.Biases.SubtractToSelf((FirstMomentBiases, SecondMomentBiases, GradientCostBiases).Map(WeightReduction));

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
}
