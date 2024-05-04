using MachineLearning.Model.Embedding;
using MachineLearning.Model.Layer;

namespace MachineLearning.Model;

public sealed class SimpleNetwork<TInput, TOutput, TLayer>(TLayer[] layers, IEmbedder<TInput, Number[], TOutput> embedder) : INetwork<TInput, Number, TOutput, TLayer> where TLayer : ILayer<Number>
{
    public TLayer[] Layers { get; } = layers;
    public TLayer OutputLayer => Layers[^1];
    public IEmbedder<TInput, Number[], TOutput> Embedder { get; } = embedder;

    public TOutput Process(TInput input)
    {
        var weights = Embedder.Embed(input);
        foreach (var layer in Layers)
        {
            weights = layer.Process(weights);
        }
        return Embedder.UnEmbed(weights);
    }

    public static INetwork<TInput, Number, TOutput, TLayer> Create(TLayer[] layers, IEmbedder<TInput, Number[], TOutput> embedder)
    {
        return new SimpleNetwork<TInput, TOutput, TLayer>(layers, embedder);
    }
}
