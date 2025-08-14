using System.Collections.Generic;

namespace ML.Analyzer.LayerFile.Operations;

internal sealed class DotProductOperation(Weights left, Weights right, Weights result) : Operation
{
    public Weights Left { get; } = left.Type is NumberType.Vector ? left : throw new InvalidOperationException();
    public Weights Right { get; } = right.Type is NumberType.Vector ? right : throw new InvalidOperationException();
    public override Weights Result { get; } = result.Type is NumberType.Single ? result : throw new InvalidOperationException();

    public override void AppendCode(StringBuilder sb)
    {
        sb.AppendLine($$"""{{Result.PassAccess()}} = {{Left.PassAccess()}}.Dot({{Right.PassAccess()}});""");
    }

    public override void AppendGradientOp(List<Operation> ops, LayerRegistry registry)
    {
        throw new NotImplementedException();
    }
}