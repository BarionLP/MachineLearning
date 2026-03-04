namespace ML.Core.Evaluation;

public sealed class TrainingEvaluationResult
{
    public required TrainingEvaluationContext Context { get; init; }
    public required EvaluationResult Result { get; init; }
    public TimeSpan Duration { get; init; }
    public override string ToString() => $"{Result.CorrectPercentage*100,5:F1}% | {Result.AverageCost:F4} | {Result.TotalElapsedTime:ss\\.ff}s ({Result.AverageElapsedTime:ss\\.ff}s) | {Context} | {Result.AverageCount}";

    public static string GetHeader() => "  ✅   | Cost   | Time   (/batch) | epoch   batch   | entries";
}
