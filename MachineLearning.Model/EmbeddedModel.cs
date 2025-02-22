using MachineLearning.Model.Layer;

namespace MachineLearning.Model;

public sealed class EmbeddedModel<TIn, TOut> : IEmbeddedModel<TIn, TOut>
{
    public required IEmbeddingLayer<TIn> InputLayer { get; init; }
    public required FeedForwardModel InnerModel { get; init; }
    public required IUnembeddingLayer<TOut> OutputLayer { get; init; }

    public long WeightCount => InputLayer.WeightCount + InnerModel.WeightCount + OutputLayer.WeightCount;

    public (TOut prediction, Weight confidence) Process(TIn input) 
        => OutputLayer.Process(InnerModel.Process(InputLayer.Process(input)));
    
    public override string ToString() => $"Embedded {InnerModel}";
}