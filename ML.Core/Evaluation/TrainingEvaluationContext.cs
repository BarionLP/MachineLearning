namespace ML.Core.Evaluation;

public sealed class TrainingEvaluationContext
{
    public required int CurrentEpoch { get; init; }
    public required int MaxEpoch { get; init; }
    public required int CurrentBatch { get; init; }
    public required int MaxBatch { get; init; }
    public required double LearningRate { get; init; }
    public override string ToString() => $"{CurrentEpoch,2}/{MaxEpoch,-2} {CurrentBatch,4}/{MaxBatch,-4}";
}
