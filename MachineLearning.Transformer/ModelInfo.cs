using MachineLearning.Model.Initialization;

namespace MachineLearning.Transformer;

public sealed record ModelInfo
{
    public string ValidTokens { get; init; } = " abcdefghijklmnopqrstuvwxyz.";
    public int TokenCount => ValidTokens.Length;
    public required int EmbeddingDimensions { get; init; }
    public required int ContextSize { get; init; }
    public required int KeyQueryDimensions { get; init; }
    public required int AttentionHeadCountPerBlock { get; init; }
    public required int AttentionBlockCount { get; init; }
    public required IInitializer Initializer { get; init; }
    public float Temperature { get; init; } = 1.5f;
}
