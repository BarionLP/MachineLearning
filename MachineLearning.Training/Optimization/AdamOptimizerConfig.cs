using MachineLearning.Training.Cost;

namespace MachineLearning.Training.Optimization;

public sealed record AdamOptimizerConfig : IOptimizerConfig
{
    public required Weight LearningRate { get; init; }
    public Weight FirstDecayRate { get; init; } = 0.9;
    public Weight SecondDecayRate { get; init; } = 0.99; //or 0.999?
    public Weight Epsilon { get; init; } = 1e-8;
    public required ICostFunction CostFunction { get; init; }

    public IOptimizer CreateOptimizer() => new AdamOptimizer(this);
}
