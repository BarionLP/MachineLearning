using MachineLearning.Training.Cost;

namespace MachineLearning.Training.Optimization;

public sealed record AdamOptimizerConfig : IOptimizerConfig<double>
{
    public required double LearningRate { get; init; }
    public double FirstDecayRate { get; init; } = 0.9;
    public double SecondDecayRate { get; init; } = 0.99; //or 0.999?
    public double Epsilon { get; init; } = 1e-8;
    public required ICostFunction CostFunction { get; init; }

    public IOptimizer<double> CreateOptimizer() => new AdamOptimizer(this);
}
