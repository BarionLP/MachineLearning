using System.Collections.Generic;

namespace ML.Analyzer.LayerFile.Operations;

internal sealed class DefineOperation(Weights weights, Weights result) : Operation
{
    public Weights Weights { get; } = weights;
    public override Weights Result { get; } = result;

    public override void AppendCode(MethodBodyWriter sb)
    {
        sb.WriteOperation($$"""var {{Result.PassAccess()}} = {{Weights.PassAccess()}};""");
    }

    public override void AppendGradientOp(List<Operation> ops, LayerRegistry registry, OperationFactory factory)
    {
        throw new NotImplementedException();
    }
}
