namespace MachineLearning.Model.Layer.Initialization;

public interface ILayerInitializer<TWeights> where TWeights : struct, IEquatable<TWeights>, IFormattable
{
    void Initialize(Matrix<TWeights> weights, Vector<TWeights> biases);
}
