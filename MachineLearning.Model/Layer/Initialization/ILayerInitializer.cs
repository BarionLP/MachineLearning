namespace MachineLearning.Model.Layer.Initialization;

public interface ILayerInitializer<TWeights>
{
    void Initialize(TWeights[,] weights, TWeights[] biases);
}
