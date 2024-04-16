using Simple.Network.Embedding;
using Simple.Network.Layer;

namespace Simple.Network;

public sealed class SimpleNetwork<TInput, TOutput>(ILayer<Number>[] layers, IEmbedder<TInput, double[], TOutput> embedder) : INetwork<TInput, Number, TOutput, ILayer<Number>> {
    public ILayer<Number>[] Layers { get; } = layers;
    public ILayer<Number> OutputLayer => Layers[^1];
    public IEmbedder<TInput, Number[], TOutput> Embedder { get; } = embedder;

    public TOutput Process(TInput input) {
        var weights = Embedder.Embed(input);
        foreach(var layer in Layers) {
            weights = layer.Process(weights);
        }
        return Embedder.UnEmbed(weights);
    }
}
