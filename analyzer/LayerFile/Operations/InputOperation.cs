using System.Collections.Generic;

namespace ML.Analyzer.LayerFile.Operations;

internal sealed class InputOperation(Weights weights, Weights result) : Operation
{
    public Weights Weights { get; } = weights;
    public override Weights Result { get; } = result;

    public override void AppendCode(StringBuilder sb)
    {
        sb.AppendLine($$"""{{Weights.PassAccess()}}.CopyTo({{Result.PassAccess()}});""");
    }

    public override void AppendGradientOp(List<Operation> ops, LayerRegistry registry)
    {
        ops.Add(new OutputOperation(registry.GetGradient(Result)));
    }
}
