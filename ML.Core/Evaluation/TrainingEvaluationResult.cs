namespace ML.Core.Evaluation;

public sealed class TrainingEvaluationResult
{
    public required TrainingEvaluationContext Context { get; init; }
    public required EvaluationResult Result { get; init; }
    public TimeSpan Duration { get; init; }
    public override string ToString() => $"{Result.ToColoredString()} | {Result.AverageCost:F4} | {Result.TotalElapsedTime:ss\\.ff}s ({Result.AverageElapsedTime:ss\\.ff}s) | {Context} | {Result.AverageCount}";

    // Emoji helps quickly finding the start of the current training run
    public static string GetHeader() => $"{EvaluationResult.GetHeader()}   | Time   (/batch) | epoch   batch   | entries";
}
