using System.Text;

namespace MachineLearning.Training.Evaluation;

public sealed class ModelTrainingResult
{
    public required int EpochCount { get; init; }
    public required ModelEvaluationResult Before { get; init; }
    public required ModelEvaluationResult After { get; init; }

    public string DumpShort()
    {
        var sb = new StringBuilder();
        sb.Append("Training Results: ")
        .Append(Before.DumpCorrectPrecentages())
        .Append(" -> ")
        .Append(After.DumpCorrectPrecentages());

        return sb.ToString();
    }
}
