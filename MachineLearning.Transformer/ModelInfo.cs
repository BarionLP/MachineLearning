namespace MachineLearning.Transformer;

public sealed record ModelInfo
{
    public string ValidTokens { get; init; } = " abcdefghijklmnopqrstuvwxyz.";
    public int TokenCount => ValidTokens.Length;
    public required int EmbeddingDimensions { get; init; }
    public required int ContextSize { get; init; }
    public required int QueryDimensions { get; init; }
    public required int AttentionHeadCount { get; init; }
    public double Temperature { get; init; } = 1.5;
}
