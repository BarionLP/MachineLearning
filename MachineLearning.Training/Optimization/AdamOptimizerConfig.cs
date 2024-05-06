using MachineLearning.Training.Cost;

namespace MachineLearning.Training.Optimization;

public sealed record AdamOptimizerConfig : IOptimizerConfig<Number>
{
    public required Number LearningRate { get; init; }
    public Number FirstDecayRate { get; init; } = 0.9;
    public Number SecondDecayRate { get; init; } = 0.99; //or 0.999?
    public Number Epsilon { get; init; } = 1e-8;
    public required ICostFunction CostFunction { get; init; }

    public IOptimizer<double> CreateOptimizer() => new AdamOptimizer(this);
}
