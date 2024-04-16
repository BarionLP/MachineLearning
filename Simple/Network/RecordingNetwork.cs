using Simple.Network.Activation;
using Simple.Network.Embedding;
using Simple.Network.Layer;

namespace Simple.Network;

public sealed class RecordingNetwork<TInput, TOutput> : INetwork<TInput, Number, TOutput, RecordingLayer>{
    public RecordingLayer[] Layers { get;  }
    public RecordingLayer OutputLayer => Layers[^1];
    public IActivation ActivationMethod { get; init; } = SigmoidActivation.Instance;
    public required IEmbedder<TInput, Number[], TOutput> Embedder { get; init; }

    public Number[] LastOutputWeights { get; private set; } = []; 

    public RecordingNetwork(params int[] layerSizes){
        Layers = new RecordingLayer[layerSizes.Length - 1]; // no need to define the input layer
        foreach (var i in ..Layers.Length){
            Layers[i] = new RecordingLayer(layerSizes[i], layerSizes[i + 1]) { ActivationMethod = ActivationMethod };
        }
    }

    public TOutput Process(TInput input) {
        var weights = Embedder.Embed(input);
        foreach (var layer in Layers){
            weights = layer.Process(weights);
        }
        LastOutputWeights = weights;
        return Embedder.UnEmbed(weights);
    }
}