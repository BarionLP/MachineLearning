namespace Simple.Training.Evaluation;

public sealed class NetworkEvaluation {
    public required NetworkEvaluationContext Context { get; init; }
    public required NetworkEvaluationResult Result { get; init; }
    public string Dump() => $"Correct: {Result.DumpCorrectPrecentages()}\t({Context.Dump()})";
}
