using System.Collections.Generic;

namespace ML.Analyzer.LayerFile.Operations;

internal sealed class ConditionOperation(string condition, Operation whenTrue) : Operation
{
    public string Condition { get; } = condition;
    public Operation WhenTrue { get; } = whenTrue;
    public override Weights Result => WhenTrue.Result;

    public override void AppendCode(MethodBodyWriter sb)
    {
        sb.WriteOperation($"if ({Condition})");
        sb.WriteOperation("{");
        sb.Indent++;
        WhenTrue.AppendCode(sb);
        sb.Indent--;
        sb.WriteOperation("}");
    }

    public override void AppendGradientOp(List<Operation> ops, LayerRegistry registry, OperationFactory factory)
    {
        throw new NotImplementedException($"cannot reverse a conditional");
    }
}