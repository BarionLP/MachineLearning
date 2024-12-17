using MachineLearning.Model.Activation;
using MachineLearning.Model.Embedding;
using MachineLearning.Model.Layer;
using MachineLearning.Model.Layer.Initialization;
using System.Collections.Immutable;
using System.Diagnostics;

namespace MachineLearning.Model;

public sealed class ModelBuilder(int inputNodeCount)
{
    private List<LayerFactory> Layers { get; } = [];
    public int InputNodeCount { get; } = inputNodeCount;
    public IActivationFunction DefaultActivationFunction { get; set; } = SigmoidActivation.Instance;

    public ModelBuilder DefaultActivation(IActivationFunction activationMethod)
    {
        DefaultActivationFunction = activationMethod;
        return this;
    }
    public ModelBuilder AddLayer(int nodeCount, IInitializer<FeedForwardLayer> initializer, IActivationFunction? activationMethod = null)
    {
        Layers.Add(
            new LayerFactory(Layers.Count == 0 ? InputNodeCount : Layers[^1].OutputNodeCount, nodeCount)
            .SetActivationFunction(activationMethod ?? DefaultActivationFunction).SetInitializer(initializer)
        );
        return this;
    }
    public ModelBuilder AddLayer(int nodeCount, Action<LayerFactory> consumer)
    {
        var layerBuilder = new LayerFactory(Layers.Count == 0 ? InputNodeCount : Layers[^1].OutputNodeCount, nodeCount)
            .SetActivationFunction(DefaultActivationFunction);
        consumer.Invoke(layerBuilder);
        Layers.Add(layerBuilder);
        return this;
    }

    public FeedForwardModel Build() => new() { Layers = Layers.Select(l => l.Create()).ToImmutableArray() };

    public EmbeddedModel<TInput, TOutput> Build<TInput, TOutput>(IEmbedder<TInput, TOutput> embedder) => new() 
    { 
        InputLayer = embedder,
        InnerModel = Build(),
        OutputLayer = embedder,
    };
}

public static class AdvancedModelBuilder
{

    public static HiddenLayerConfig<TInput> Create<TLayer, TInput>(TLayer layer, IInitializer<TLayer> initializer) where TLayer : IEmbeddingLayer<TInput>
    {
        initializer.Initialize(layer);
        return Create(layer);
    }

    public static HiddenLayerConfig<TInput> Create<TInput>(IEmbeddingLayer<TInput> layer)
        => new(layer);

    public sealed class HiddenLayerConfig<TInput>(IEmbeddingLayer<TInput> layer)
    {
        public IEmbeddingLayer<TInput> InputLayer { get; } = layer;
        public List<LayerFactory> Layers { get; } = [];
        public IActivationFunction DefaultActivationFunction { get; set; } = SigmoidActivation.Instance;
        public int LastOutputNodeCount = layer.OutputNodeCount;

        public HiddenLayerConfig<TInput> SetDefaultActivationFunction(IActivationFunction activationFunction)
        {
            DefaultActivationFunction = activationFunction;
            return this;
        }
        public HiddenLayerConfig<TInput> AddLayer(int nodeCount, IInitializer<FeedForwardLayer> initializer, IActivationFunction? activationMethod = null)
        {
            return AddLayer(
                new LayerFactory(LastOutputNodeCount, nodeCount)
                .SetActivationFunction(activationMethod ?? DefaultActivationFunction).SetInitializer(initializer)
            );
        }
        public HiddenLayerConfig<TInput> AddLayer(int nodeCount, Action<LayerFactory> consumer)
        {
            var layerBuilder = new LayerFactory(LastOutputNodeCount, nodeCount)
                .SetActivationFunction(DefaultActivationFunction);
            consumer.Invoke(layerBuilder);
            return AddLayer(layerBuilder);
        }

        private HiddenLayerConfig<TInput> AddLayer(LayerFactory layerFactory)
        {
            LastOutputNodeCount = layerFactory.OutputNodeCount;
            Layers.Add(layerFactory);
            return this;
        }

        public EmbeddedModel<TInput, TOutput> AddOutputLayer<TOutput>(Func<int, IUnembeddingLayer<TOutput>> outputLayer)
            => AddOutputLayer(outputLayer(LastOutputNodeCount));

        public EmbeddedModel<TInput, TOutput> AddOutputLayer<TOutput>(IUnembeddingLayer<TOutput> outputLayer)
        {
            Debug.Assert(LastOutputNodeCount == outputLayer.InputNodeCount);
            return new()
            {
                InputLayer = InputLayer,
                InnerModel = new() { Layers = Layers.Select(l => l.Create()).ToImmutableArray() },
                OutputLayer = outputLayer,
            };
        }
    }
}