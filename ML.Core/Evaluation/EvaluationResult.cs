namespace ML.Core.Evaluation;

public sealed class EvaluationResult
{
    public static readonly EvaluationResult ZERO = new() { TotalCount = 0, CorrectCount = 0, CorrectConfidenceSum = 0, WrongConfidenceSum = 0, TotalCost = 0, TotalElapsedTime = TimeSpan.Zero, stackCount = 0 };
    public required int TotalCount { get; init; }
    public int AverageCount => TotalCount / stackCount;
    public required int CorrectCount { get; init; }
    public float CorrectPercentage => (float)CorrectCount / TotalCount;
    public int WrongCount => TotalCount - CorrectCount;
    public float WrongPercentage => (float)WrongCount / TotalCount;

    public required float CorrectConfidenceSum { get; init; }
    public float CorrectConfidence => CorrectConfidenceSum / CorrectCount;

    public required float WrongConfidenceSum { get; init; }
    public float WrongConfidence => WrongConfidenceSum / WrongCount;

    public required double TotalCost { get; init; }
    public double AverageCost => TotalCost / TotalCount;

    public TimeSpan TotalElapsedTime { get; init; } = TimeSpan.Zero;
    public TimeSpan AverageElapsedTime => TotalElapsedTime / stackCount;
    private int stackCount = 1;

    public static EvaluationResult operator +(EvaluationResult left, EvaluationResult right) => new()
    {
        TotalCount = left.TotalCount + right.TotalCount,
        CorrectCount = left.CorrectCount + right.CorrectCount,
        CorrectConfidenceSum = left.CorrectConfidenceSum + right.CorrectConfidenceSum,
        WrongConfidenceSum = left.WrongConfidenceSum + right.WrongConfidenceSum,
        TotalCost = left.TotalCost + right.TotalCost,
        TotalElapsedTime = left.TotalElapsedTime + right.TotalElapsedTime,
        stackCount = left.stackCount + right.stackCount
    };

    public override string ToString() => $"{CorrectPercentage * 100,5:F1}% | {CorrectConfidence:F2} {WrongConfidence:F2}";
    public string ToColoredString() => $"{ConfidenceToTextColor(CorrectPercentage)}{CorrectPercentage * 100,5:F1}%{RESET_COLOR} | {CorrectConfidence:F2} {WrongConfidence:F2}";

    public static string GetHeader() => "  ✅   | Conf.     | Cost";

    const string RESET_COLOR = "\u001b[0m";
    static string ConfidenceToTextColor(Weight confidence) => $"\u001b[38;2;{(1 - confidence) * 255:F0};{confidence * 255:F0};60m";
}
