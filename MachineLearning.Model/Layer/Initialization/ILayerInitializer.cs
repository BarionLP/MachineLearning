namespace MachineLearning.Model.Layer.Initialization;

public interface ILayerInitializer
{
    void Initialize(Matrix weights, Vector biases);
}
