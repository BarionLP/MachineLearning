/* using MachineLearning.Training.Cost;

namespace MachineLearning.Training.Optimization;

public sealed class GDMomentumOptimizerConfig : IOptimizerConfig<double>
{
    public required double LearningRate { get; init; } = 0.7;
    public double LearningRateEpochMultiplier { get; init; } = 1;
    public required double Momentum { get; init; } = 0.85;
    public required double Regularization { get; init; } = 0.01;
    public ICostFunction CostFunction { get; init; } = MeanSquaredErrorCost.Instance;

    public IOptimizer<double> CreateOptimizer() => new GDMomentumOptimizer(this);
}
 */