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

public sealed class NoInitializer<T> : ILayerInitializer<T> where T : ILayer
{
    public static NoInitializer<T> Instance { get; } = new();
    public void Initialize(T layer) { }
} 