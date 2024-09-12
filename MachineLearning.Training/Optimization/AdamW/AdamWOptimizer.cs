using MachineLearning.Model.Layer;
using MachineLearning.Training.Cost;

namespace MachineLearning.Training.Optimization.AdamW;

public sealed class AdamWOptimizer : IGenericOptimizer
{
    public required Weight LearningRate { get; init; } = 0.1;
    public Weight FirstDecayRate { get; init; } = 0.9;
    public Weight SecondDecayRate { get; init; } = 0.99; //or 0.999?
    public Weight WeightDecayCoefficient /*λ*/ { get; init; } = 1e-3; // (1e-5 - 1e-2)
    public Weight Epsilon { get; init; } = 1e-8;
    public required ICostFunction CostFunction { get; init; }

    public double Iteration { get; set; } = 1; //(even when retraining!) when starting with 0 gradient estimates shoot to infinity?

    public void OnBatchCompleted()
    {
        Iteration++;
    }
    public ILayerOptimizer CreateLayerOptimizer(ILayer layer) => layer switch
    {
        SimpleLayer simpleLayer => new AdamWLayerOptimizer(this, simpleLayer),
        _ => throw new NotImplementedException($"No Nadam implementation for {layer}"),
    };
}