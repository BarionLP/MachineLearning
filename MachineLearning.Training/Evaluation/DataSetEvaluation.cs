namespace MachineLearning.Training.Evaluation;

public sealed class DataSetEvaluation
{
    public required TrainingEvaluationContext Context { get; init; }
    public required DataSetEvaluationResult Result { get; init; }
    public string Dump() => $"Correct: {Result.CorrectPercentage:P0}\t({Context.Dump()}\tbatchSize {Result.TotalCount})";
}
