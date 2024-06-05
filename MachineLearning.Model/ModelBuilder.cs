using MachineLearning.Domain.Activation;
using MachineLearning.Model.Embedding;
using MachineLearning.Model.Layer;
using MachineLearning.Model.Layer.Initialization;

namespace MachineLearning.Model;

public sealed class ModelBuilder<TNetwork, TInput, TOutput, TLayer>(int inputNodeCount) where TNetwork : INetwork<TInput, TOutput, TLayer> where TLayer : ILayer
{
    private List<LayerBuilder<TLayer>> Layers { get; } = [];
    public int InputNodeCount { get; } = inputNodeCount;

    public IEmbedder<TInput, TOutput>? Embedder { get; set; }
    public IActivationMethod DefaultActivationMethod { get; set; } = SigmoidActivation.Instance;

    public ModelBuilder<TNetwork, TInput, TOutput, TLayer> SetEmbedder(IEmbedder<TInput, TOutput> embedder)
    {
        Embedder = embedder;
        return this;
    }
    public ModelBuilder<TNetwork, TInput, TOutput, TLayer> SetDefaultActivationMethod(IActivationMethod activationMethod)
    {
        DefaultActivationMethod = activationMethod;
        return this;
    }
    public ModelBuilder<TNetwork, TInput, TOutput, TLayer> AddLayer(int nodeCount)
    {
        Layers.Add(
            new LayerBuilder<TLayer>(Layers.Count == 0 ? InputNodeCount : Layers[^1].OutputNodeCount, nodeCount)
            .SetActivationMethod(DefaultActivationMethod)
        );
        return this;
    }
    public ModelBuilder<TNetwork, TInput, TOutput, TLayer> AddLayer(int nodeCount, ILayerInitializer initializer)
    {
        Layers.Add(
            new LayerBuilder<TLayer>(Layers.Count == 0 ? InputNodeCount : Layers[^1].OutputNodeCount, nodeCount)
            .SetActivationMethod(DefaultActivationMethod).Initialize(initializer)
        );
        return this;
    }
    public ModelBuilder<TNetwork, TInput, TOutput, TLayer> AddLayer(int nodeCount, Action<LayerBuilder<TLayer>> consumer)
    {
        var layerBuilder = new LayerBuilder<TLayer>(Layers.Count == 0 ? InputNodeCount : Layers[^1].OutputNodeCount, nodeCount)
            .SetActivationMethod(DefaultActivationMethod);
        consumer.Invoke(layerBuilder);
        Layers.Add(layerBuilder);
        return this;
    }

    public TNetwork Build()
    {
        return (TNetwork) TNetwork.Create(Layers.Select(l => l.Build()).ToArray(), Embedder ?? throw new NullReferenceException("NetworkBuilder needs an embedder!"));
    }
}

public static class ModelBuilder
{
    public static ModelBuilder<SimpleNetwork<TInput, TOutput, RecordingLayer>, TInput, TOutput, RecordingLayer> Recorded<TInput, TOutput>(int inputNodeCount) => new(inputNodeCount);
    public static ModelBuilder<SimpleNetwork<TInput, TOutput, SimpleLayer>, TInput, TOutput, SimpleLayer> Simple<TInput, TOutput>(int inputNodeCount) => new(inputNodeCount);
}