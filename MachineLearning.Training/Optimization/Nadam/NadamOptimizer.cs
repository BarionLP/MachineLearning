using MachineLearning.Model.Embedding;
using MachineLearning.Model.Layer;
using MachineLearning.Training.Optimization.Adam;

namespace MachineLearning.Training.Optimization.Nadam;

public sealed class NadamOptimizer : AdamOptimizer
{
    public override ILayerOptimizer CreateLayerOptimizer(ILayer layer) => layer switch
    {
        FeedForwardLayer simpleLayer => new SimpleNadamOptimizer(this, simpleLayer),
        StringEmbeddingLayer stringLayer => new StringNadamOptimizer(this, stringLayer),
        IEmbedder<string, char> or TokenOutputLayer => new EmptyAdamOptimizer(layer),
        _ => throw new NotImplementedException($"No Nadam implementation for {layer}"),
    };

    
}
