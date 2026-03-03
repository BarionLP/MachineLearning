namespace ML.Core.Evaluation;

public sealed class TrainingEvaluationContext
{
    public required int CurrentEpoch { get; init; }
    public required int MaxEpoch { get; init; }
    public required int CurrentBatch { get; init; }
    public required int MaxBatch { get; init; }
    public required double LearningRate { get; init; }
    public override string ToString() => $"epoch {CurrentEpoch}/{MaxEpoch}\tbatch {CurrentBatch}/{MaxBatch}";
}
