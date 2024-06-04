using MachineLearning.Model.Embedding;
using MachineLearning.Model.Layer;

namespace MachineLearning.Model;

public sealed class SimpleNetwork<TInput, TOutput, TLayer>(TLayer[] layers, IEmbedder<TInput, TOutput> embedder) : INetwork<TInput, TOutput, TLayer> where TLayer : ILayer
{
    public TLayer[] Layers { get; } = layers;
    public TLayer OutputLayer => Layers[^1];
    public IEmbedder<TInput, TOutput> Embedder { get; } = embedder;

    public TOutput Process(TInput input)
    {
        var weights = Forward(Embedder.Embed(input));
        return Embedder.UnEmbed(weights);
    }
    public Vector Forward(Vector weights)
    {
        foreach(var layer in Layers)
        {
            weights = layer.Forward(weights);
        }
        return weights;
    }

    public static INetwork<TInput, TOutput, TLayer> Create(TLayer[] layers, IEmbedder<TInput, TOutput> embedder)
    {
        return new SimpleNetwork<TInput, TOutput, TLayer>(layers, embedder);
    }
}
