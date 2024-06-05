using System.Collections.Immutable;

namespace BarionGPT;

public sealed class Model(ModelInfo info)
{
    public ModelInfo Info { get; } = info;

    private readonly Embedder _embedder = new(info);
    private readonly ImmutableArray<AttentionHead> _attentionHeads = CreateAttentionHeads(info);
    private readonly Unembedder _unembedder = new(info);

    public char Process(string input)
    {
        var embedding = _embedder.Embed(input);
        var result = embedding.Clone();
        foreach(var delta in _attentionHeads/*.AsParallel()*/.Select(head => head.GetEmbeddingDelta(embedding)))
        {
            result += delta;
        }
        return _unembedder.Unembed(result);
    }

    private static ImmutableArray<AttentionHead> CreateAttentionHeads(ModelInfo info)
    {
        return Enumerable.Range(0, info.AttentionHeadCount).Select(_ => new AttentionHead(info)).ToImmutableArray();
    }
}
