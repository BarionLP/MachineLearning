using MachineLearning.Domain;
using System.Collections.Immutable;

namespace MachineLearning.Model.Embedding;

public sealed class MNISTEmbedder(ImmutableArray<int> _nodeMapping) : IEmbedder<Vector, int>
{
    public static MNISTEmbedder Instance { get; } = new MNISTEmbedder([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);

    private readonly ImmutableArray<int> _nodeMapping = _nodeMapping;

    public Vector Embed(Vector input) => input;
    public int UnEmbed(Vector input) => _nodeMapping[input.MaximumIndex()];
}
