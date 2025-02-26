using System.Collections.Immutable;
using MachineLearning.Model.Layer;
using MachineLearning.Model.Layer.Snapshot;

namespace MachineLearning.Samples.MNIST;

public sealed class MNISTEmbedder() : IEmbeddingLayer<double[]>
{
    public static MNISTEmbedder Instance { get; } = new MNISTEmbedder();

    public Vector Process(double[] input) => Vector.Of([.. input.Select(x => (float)x)]);

    public Vector Process(double[] input, ILayerSnapshot _) => Process(input);

    int IEmbeddingLayer<double[]>.OutputNodeCount => 784;
    long ILayer.WeightCount => 0;
    ILayerSnapshot ILayer.CreateSnapshot() => LayerSnapshots.Empty;
    public IGradients CreateGradientAccumulator() => IGradients.Empty;
}

public sealed class MNISTUnEmbedder(ImmutableArray<int> _nodeMapping) : IUnembeddingLayer<int>
{
    public static MNISTUnEmbedder Instance { get; } = new MNISTUnEmbedder([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);

    private readonly ImmutableArray<int> _nodeMapping = _nodeMapping;

    public (int output, Weight confidence) Process(Vector input)
    {
        var index = input.MaximumIndex();
        return (_nodeMapping[index], input[index]);
    }

    public (int output, float confidence, Vector weights) Process(Vector input, ILayerSnapshot _)
    {
        var index = input.MaximumIndex();
        return (_nodeMapping[index], input[index], input);
    }

    int IUnembeddingLayer<int>.InputNodeCount => 10;
    long ILayer.WeightCount => 0;
    ILayerSnapshot ILayer.CreateSnapshot() => LayerSnapshots.Empty;
    public IGradients CreateGradientAccumulator() => IGradients.Empty;
}
