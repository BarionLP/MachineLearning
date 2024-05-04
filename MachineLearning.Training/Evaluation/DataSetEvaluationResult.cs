namespace MachineLearning.Training.Evaluation;

public sealed class DataSetEvaluationResult
{
    public required int TotalCount { get; init; }
    public required int CorrectCount { get; init; }
    public float CorrectPercentage => (float)CorrectCount / TotalCount;
    public required Number TotalCost { get; init; }
    public Number AverageCost => TotalCost / TotalCount;
}
