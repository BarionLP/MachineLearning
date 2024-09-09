using MachineLearning.Model.Activation;
using MachineLearning.Model.Embedding;
using MachineLearning.Model.Layer;
using MachineLearning.Model.Layer.Initialization;
using System.Collections.Immutable;

namespace MachineLearning.Model;

public sealed class ModelBuilder(int inputNodeCount)
{
    private List<LayerBuilder> Layers { get; } = [];
    public int InputNodeCount { get; } = inputNodeCount;
    public IActivationMethod DefaultActivationMethod { get; set; } = SigmoidActivation.Instance;

    public ModelBuilder SetDefaultActivationMethod(IActivationMethod activationMethod)
    {
        DefaultActivationMethod = activationMethod;
        return this;
    }
    public ModelBuilder AddLayer(int nodeCount, ILayerInitializer initializer, IActivationMethod? activationMethod = null)
    {
        Layers.Add(
            new LayerBuilder(Layers.Count == 0 ? InputNodeCount : Layers[^1].OutputNodeCount, nodeCount)
            .SetActivationMethod(activationMethod ?? DefaultActivationMethod).Initialize(initializer)
        );
        return this;
    }
    public ModelBuilder AddLayer(int nodeCount, Action<LayerBuilder> consumer)
    {
        var layerBuilder = new LayerBuilder(Layers.Count == 0 ? InputNodeCount : Layers[^1].OutputNodeCount, nodeCount)
            .SetActivationMethod(DefaultActivationMethod);
        consumer.Invoke(layerBuilder);
        Layers.Add(layerBuilder);
        return this;
    }

    public SimpleModel Build() => new (Layers.Select(l => l.Build()).ToImmutableArray());

    public EmbeddedModel<TInput, TOutput> Build<TInput, TOutput>(IEmbedder<TInput, TOutput> embedder) => new (Build(), embedder);
}

//public static class ModelBuilder
//{
//    public static ModelBuilder<SimpleNetwork<TInput, TOutput, RecordingLayer>, TInput, TOutput, RecordingLayer> Recorded<TInput, TOutput>(int inputNodeCount) => new(inputNodeCount);
//    public static ModelBuilder<SimpleNetwork<TInput, TOutput, SimpleLayer>, TInput, TOutput, SimpleLayer> Simple<TInput, TOutput>(int inputNodeCount) => new(inputNodeCount);
//}