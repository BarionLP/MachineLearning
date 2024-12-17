using MachineLearning.Model.Layer;

namespace MachineLearning.Model;

public sealed class EmbeddedModel<TIn, TOut> : IModel
{
    public required IEmbeddingLayer<TIn> InputLayer { get; init; }
    public required FeedForwardModel InnerModel { get; init; }
    public required IUnembeddingLayer<TOut> OutputLayer { get; init; }

    public long ParameterCount => InputLayer.ParameterCount + InnerModel.ParameterCount + OutputLayer.ParameterCount;

    public (TOut prediction, Weight confidence) Process(TIn input) 
        => OutputLayer.Forward(InnerModel.Process(InputLayer.Forward(input)));

    public override string ToString() => $"Embedded {InnerModel}";
}