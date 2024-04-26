using System.Text;

namespace Simple.Training.Evaluation;

public sealed class NetworkTrainingResult {
    public required int EpochCount { get; init; }
    public required NetworkEvaluationResult Before { get; init; }
    public required NetworkEvaluationResult After { get; init; }
    
    public string DumpShort(){
        var sb = new StringBuilder();
        sb.Append("Training Results: ")
        .Append(Before.DumpCorrectPrecentages())
        .Append(" -> ")
        .Append(After.DumpCorrectPrecentages());

        return sb.ToString();
    }
}
