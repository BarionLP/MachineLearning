using MachineLearning.Model.Initialization;

namespace MachineLearning.Model.Layer.Initialization;

public interface ILayerInitializer : ILayerInitializer<SimpleLayer>
{
    void Initialize(Matrix weights, Vector biases);

    void ILayerInitializer<SimpleLayer>.Initialize(SimpleLayer layer) {
        Initialize(layer.Weights, layer.Biases);
    }
}

public interface ILayerInitializer<TLayer> where TLayer : ILayer {
    public void Initialize(TLayer layer);
}

public sealed class ZeroInitializer<T> : ILayerInitializer<T> where T : ILayer
{
    public static ZeroInitializer<T> Instance { get; } = new();
    public void Initialize(T layer) { }
} 