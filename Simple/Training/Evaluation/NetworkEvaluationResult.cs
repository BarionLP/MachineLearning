namespace Simple.Training.Evaluation;

public sealed class NetworkEvaluationResult {
    public required DataSetEvaluationResult TrainingSetResult { get; init; }
    public required DataSetEvaluationResult TestSetResult { get; init; }
    public string DumpCorrectPrecentages() => $"{TrainingSetResult.CorrectPercentage:P} | {TestSetResult.CorrectPercentage:P}";
}