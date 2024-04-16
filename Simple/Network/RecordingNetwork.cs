using Simple.Network.Activation;
using Simple.Network.Embedding;
using Simple.Network.Layer;

namespace Simple.Network;

public sealed class RecordingNetwork<TInput, TOutput>(RecordingLayer[] layers, IEmbedder<TInput, Number[], TOutput> embedder) : INetwork<TInput, Number, TOutput, RecordingLayer>{
    public RecordingLayer[] Layers { get;  } = layers;
    public RecordingLayer OutputLayer => Layers[^1];
    public IEmbedder<TInput, Number[], TOutput> Embedder { get; } = embedder;

    public Number[] LastOutputWeights { get; private set; } = [];

    public TOutput Process(TInput input) {
        var weights = Embedder.Embed(input);
        foreach (var layer in Layers){
            weights = layer.Process(weights);
        }
        LastOutputWeights = weights;
        return Embedder.UnEmbed(weights);
    }

    public static INetwork<TInput, Number, TOutput, RecordingLayer> Create(RecordingLayer[] layers, IEmbedder<TInput, Number[], TOutput> embedder)
        => new RecordingNetwork<TInput, TOutput>(layers, embedder);
}