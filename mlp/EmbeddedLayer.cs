using MachineLearning.Model.Attributes;
using MachineLearning.Model.Layer;
using MachineLearning.Model.Layer.Snapshot;

namespace ML.MultiLayerPerceptron;

[GeneratedLayer]
public sealed partial class EmbeddedLayer<TIn, TOut> : ILayer<TIn, TOut, EmbeddedLayer<TIn, TOut>.Snapshot>
{
    [Module] public required IEmbeddingLayer<TIn> EmbeddingLayer { get; init; }
    [Module] public required ILayer<Vector, ILayerSnapshot> HiddenLayer { get; init; }
    [Module] public required IUnembeddingLayer<TOut> UnembeddingLayer { get; init; }

    public (TOut output, Weight confidence, Vector weights) Forward(TIn Input, Snapshot snapshot)
    {
        snapshot.Input = Input;
        snapshot.InputResult = EmbeddingLayer.Process(snapshot.Input, snapshot.EmbeddingLayer);
        snapshot.HiddenResult = HiddenLayer.Forward(snapshot.InputResult, snapshot.HiddenLayer);
        var output = UnembeddingLayer.Process(snapshot.HiddenResult, snapshot.UnembeddingLayer);
        return output;
    }
    public Vector Backward(Vector outputGradient, Snapshot snapshot, Gradients gradients)
    {
        // snapshot.HiddenResult1Gradient = UnembeddingLayer.Backward(outputGradient, snapshot.UnembeddingLayer, gradients.UnembeddingLayer);
        // snapshot.InputResultGradient = HiddenLayer.Backward(snapshot.HiddenResult1Gradient, snapshot.HiddenLayer, gradients.HiddenLayer);
        // snapshot.InputGradient = EmbeddingLayer.Backward(snapshot.InputResultGradient, snapshot.EmbeddingLayer, gradients.EmbeddingLayer);
        return snapshot.InputGradient;
    }


    partial class Snapshot
    {
        public Vector InputResult { get; set; }
        public Vector HiddenResult { get; set; }
        public Vector OutputResult { get; set; }
        public TIn Input { get; set; } = default;
        public Vector HiddenResult1Gradient { get; set; }
        public Vector InputResultGradient { get; set; }
        public Vector InputGradient { get; set; }
        public ILayerSnapshot EmbeddingLayer { get; } = layer.EmbeddingLayer.CreateSnapshot();
        public ILayerSnapshot HiddenLayer { get; } = layer.HiddenLayer.CreateSnapshot();
        public ILayerSnapshot UnembeddingLayer { get; } = layer.UnembeddingLayer.CreateSnapshot();
    }
}
