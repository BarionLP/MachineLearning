using Simple.Network.Activation;
using Simple.Network.Embedding;
using Simple.Network.Layer;

namespace Simple.Network; 

public sealed class NetworkBuilder<TNetwork, TInput, TOutput, TLayer>(int inputNodeCount) where TNetwork : INetwork<TInput, Number, TOutput, TLayer> where TLayer : ILayer<Number>{
    private List<TLayer> Layers { get; } = [];
    public int InputNodeCount { get; } = inputNodeCount;

    public IEmbedder<TInput, Number[], TOutput>? Embedder { get; set; }
    public IActivationMethod<Number> DefaultActivationMethod { get; set; } = SigmoidActivation.Instance;
    
    public NetworkBuilder<TNetwork, TInput, TOutput, TLayer> SetEmbedder(IEmbedder<TInput, Number[], TOutput> embedder) {
        Embedder = embedder;
        return this;
    }
    public NetworkBuilder<TNetwork, TInput, TOutput, TLayer> SetDefaultActivationMethod(IActivationMethod<Number> activationMethod) {
        DefaultActivationMethod = activationMethod;
        return this;
    }
    public NetworkBuilder<TNetwork, TInput, TOutput, TLayer> AddLayer(int nodeCount) {
        Layers.Add(
            new LayerBuilder<TLayer>(Layers.Count == 0 ? InputNodeCount : Layers[^1].OutputNodeCount, nodeCount)
            .SetActivationMethod(DefaultActivationMethod).Build()
        );
        return this;
    }
    public NetworkBuilder<TNetwork, TInput, TOutput, TLayer> AddLayer(int nodeCount, Action<LayerBuilder<TLayer>> consumer) {
        var layerBuilder = new LayerBuilder<TLayer>(Layers.Count == 0 ? InputNodeCount : Layers[^1].OutputNodeCount, nodeCount)
            .SetActivationMethod(DefaultActivationMethod);
        consumer.Invoke(layerBuilder);
        Layers.Add(layerBuilder.Build());
        return this;
    }
    public NetworkBuilder<TNetwork, TInput, TOutput, TLayer> AddRandomizedLayer(int nodeCount, Random? random = null) {
        Layers.Add(
            new LayerBuilder<TLayer>(Layers.Count == 0 ? InputNodeCount : Layers[^1].OutputNodeCount, nodeCount)
            .SetActivationMethod(DefaultActivationMethod)
            .InitializeRandom(random).Build()
        );
        return this;
    }

    public TNetwork Build() {
        return (TNetwork) TNetwork.Create([.. Layers], Embedder ?? throw new NullReferenceException("NetworkBuilder needs an embedder!"));
    }
}

public static class NetworkBuilder {
    public static NetworkBuilder<RecordingNetwork<TInput, TOutput>, TInput, TOutput, RecordingLayer> Recorded<TInput, TOutput>(int inputNodeCount) => new(inputNodeCount);
    public static NetworkBuilder<SimpleNetwork<TInput, TOutput, SimpleLayer>, TInput, TOutput, SimpleLayer> Simple<TInput, TOutput>(int inputNodeCount) => new(inputNodeCount);
}