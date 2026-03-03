namespace ML.Core.Evaluation;

public sealed class TrainingEvaluationResult
{
    public required TrainingEvaluationContext Context { get; init; }
    public required EvaluationResult Result { get; init; }
    public TimeSpan Duration { get; init; }
    public override string ToString() => $"Correct: {Result.CorrectPercentage,5:P0} ({Result.AverageCost:F4})\t({Result.AverageElapsedTime:ss\\.ff}s/batch\t{Context} ({Result.AverageCount}))";
}
