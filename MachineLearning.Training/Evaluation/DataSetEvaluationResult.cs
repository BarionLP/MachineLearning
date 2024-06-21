namespace MachineLearning.Training.Evaluation;

public sealed class DataSetEvaluationResult
{
    public static readonly DataSetEvaluationResult ZERO = new() { TotalCount = 0, CorrectCount = 0, TotalCost = 0, stackCount = 0 };
    public required int TotalCount { get; init; }
    public int AverageCount => TotalCount / stackCount;
    public required int CorrectCount { get; init; }
    public float CorrectPercentage => (float) CorrectCount / TotalCount;
    public required double TotalCost { get; init; }
    public double AverageCost => TotalCost / TotalCount;
    private int stackCount = 1;

    public static DataSetEvaluationResult operator +(DataSetEvaluationResult left, DataSetEvaluationResult right) => new()
    {
        TotalCount = left.TotalCount + right.TotalCount,
        CorrectCount = left.CorrectCount + right.CorrectCount,
        TotalCost = left.TotalCost + right.TotalCost,
        stackCount = left.stackCount + right.stackCount
    };
}
