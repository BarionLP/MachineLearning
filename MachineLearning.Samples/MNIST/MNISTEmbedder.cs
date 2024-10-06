using System.Collections.Immutable;
using MachineLearning.Model.Layer.Snapshot;

namespace MachineLearning.Samples.MNIST;

public sealed class MNISTEmbedder(ImmutableArray<int> _nodeMapping) : IEmbedder<double[], int>
{
    public static MNISTEmbedder Instance { get; } = new MNISTEmbedder([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);

    private readonly ImmutableArray<int> _nodeMapping = _nodeMapping;

    public Vector Embed(double[] input) => Vector.Of(input);
    public (int output, Weight confidence) Unembed(Vector input)
    {
        var index = input.MaximumIndex();
        return (_nodeMapping[index], input[index]);
    }

    public Vector Embed(double[] input, ILayerSnapshot snapshot)
    {
        return Embed(input);
    }

    public (int output, int index, Vector weights) Unembed(Vector input, ILayerSnapshot snapshot)
    {
        var index = input.MaximumIndex();
        return (_nodeMapping[index], index, input);
    }
}
