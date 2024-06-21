namespace MachineLearning.Training.Evaluation;

public sealed class TrainingEvaluationContext
{
    public required int CurrentEpoch { get; init; }
    public required int MaxEpoch { get; init; }
    public required int CurrentBatch { get; init; }
    public required int MaxBatch { get; init; }
    public required double LearnRate { get; init; }
    public string Dump() => $"epoch {CurrentEpoch}/{MaxEpoch}\tbatch {CurrentBatch}/{MaxBatch}";
}
