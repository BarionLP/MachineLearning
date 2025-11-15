using System.Collections.Generic;

namespace ML.Analyzer.LayerFile.Operations;

internal sealed class ActivationGradientOperation(Weights source, Weights result) : Operation
{
    public Weights Source { get; } = source;
    public override Weights Result { get; } = result;

    public override void AppendCode(MethodBodyWriter sb)
    {
        sb.WriteOperation($"ActivationFunction.DerivativeTo({Source.PassAccess()}, {Result.PassAccess()});");
    }

    public override void AppendGradientOp(List<Operation> ops, LayerRegistry registry, OperationFactory factory)
    {
        throw new NotImplementedException();
    }
}
