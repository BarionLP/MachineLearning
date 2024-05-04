namespace MachineLearning.Training.Evaluation;

public sealed class ModelEvaluationResult
{
    public required DataSetEvaluationResult TrainingSetResult { get; init; }
    public required DataSetEvaluationResult TestSetResult { get; init; }
    public string DumpCorrectPrecentages() => $"{TrainingSetResult.CorrectPercentage:P} | {TestSetResult.CorrectPercentage:P}";
}