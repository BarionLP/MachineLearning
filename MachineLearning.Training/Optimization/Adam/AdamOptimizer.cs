using MachineLearning.Model.Embedding;
using MachineLearning.Model.Layer;
using MachineLearning.Training.Cost;

namespace MachineLearning.Training.Optimization.Adam;

public class AdamOptimizer : IGenericOptimizer
{
    public required Weight LearningRate { get; init; } = 0.1f;
    public Weight FirstDecayRate { get; init; } = 0.9f;
    public Weight SecondDecayRate { get; init; } = 0.99f; //or 0.999?
    public Weight Epsilon { get; init; } = 1e-7f;
    public required ICostFunction CostFunction { get; init; }

    public Weight Iteration { get; set; } = 1; // (even when retraining!) when starting with 0 gradient estimates shoot to infinity?

    public void OnBatchCompleted()
    {
        Iteration++;
    }

    public virtual ILayerOptimizer CreateLayerOptimizer(ILayer layer) => layer switch
    {
        SimpleLayer simpleLayer => new SimpleAdamOptimizer(this, simpleLayer),
        StringEmbeddingLayer stringLayer => new StringAdamOptimizer(this, stringLayer),
        IEmbedder<string, char> or IEmbedder<float[], int> or TokenOutputLayer or EncodedEmbeddingLayer => new EmptyAdamOptimizer(layer),
        _ => throw new NotImplementedException($"No Adam implementation for {layer}"),
    };
}