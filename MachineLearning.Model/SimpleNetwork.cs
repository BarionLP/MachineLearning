using MachineLearning.Model.Embedding;
using MachineLearning.Model.Layer;

namespace MachineLearning.Model;

public sealed class SimpleNetwork<TInput, TOutput, TLayer>(TLayer[] layers, IEmbedder<TInput, Vector<double>, TOutput> embedder) : INetwork<TInput, double, TOutput, TLayer> where TLayer : ILayer<double>
{
    public TLayer[] Layers { get; } = layers;
    public TLayer OutputLayer => Layers[^1];
    public IEmbedder<TInput, Vector<double>, TOutput> Embedder { get; } = embedder;

    public TOutput Process(TInput input)
    {
        var weights = Forward(Embedder.Embed(input));
        return Embedder.UnEmbed(weights);
    }
    public Vector<double> Forward(Vector<double> weights)
    {
        foreach (var layer in Layers)
        {
            weights = layer.Forward(weights);
        }
        return weights;
    }

    public static INetwork<TInput, double, TOutput, TLayer> Create(TLayer[] layers, IEmbedder<TInput, Vector<double>, TOutput> embedder)
    {
        return new SimpleNetwork<TInput, TOutput, TLayer>(layers, embedder);
    }
}
