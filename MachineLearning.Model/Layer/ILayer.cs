namespace MachineLearning.Model.Layer;

public interface IEmbeddingLayer<in TInput> : ILayer
{
    public int OutputNodeCount { get; }

    public Vector Forward(TInput input);
}

public interface IUnembeddingLayer<TOutput> : ILayer
{
    public int InputNodeCount { get; }

    public (TOutput output, Weight confidence) Forward(Vector input);
}

public interface ILayer 
{
    public uint ParameterCount { get; }
};