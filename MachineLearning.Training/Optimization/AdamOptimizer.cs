using MachineLearning.Model.Layer;
using MachineLearning.Training.Cost;
using MachineLearning.Training.Optimization.Layer;

namespace MachineLearning.Training.Optimization;

public sealed class AdamOptimizer : IOptimizer<Number>
{
    public required Number LearningRate { get; init; }
    public Number GradientsDecayRate { get; init; } = 0.9;
    public Number SquaredGradientsDecayRate { get; init; } = 0.99; //or 0.999?
    public Number Epsilon { get; init; } = 1e-8;
    public Number Iteration { get; set; } = 1; //(even when retraining!) when starting with 0 gradient estimates shoot to infinity?
    public ICostFunction CostFunction { get; init; } = MeanSquaredErrorCost.Instance;

    public void OnBatchCompleted()
    {
        Iteration++;
    }

    public ILayerOptimizer<Number> CreateLayerOptimizer(RecordingLayer layer) => new AdamLayerOptimizer(this, layer, CostFunction);
}
