namespace MachineLearning.Training.Evaluation;

public sealed class ModelEvaluation
{
    public required TrainingEvaluationContext Context { get; init; }
    public required ModelEvaluationResult Result { get; init; }
    public string Dump() => $"Correct: {Result.DumpCorrectPrecentages()}\t({Context.Dump()})";
}
