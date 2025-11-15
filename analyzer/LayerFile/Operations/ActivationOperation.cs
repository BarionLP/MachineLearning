using System.Collections.Generic;

namespace ML.Analyzer.LayerFile.Operations;

internal sealed class ActivationOperation(Weights source, Weights result) : Operation
{
    public Weights Source { get; } = source;
    public override Weights Result { get; } = result;

    public override void AppendCode(MethodBodyWriter sb)
    {
        sb.WriteOperation($"ActivationFunction.ActivateTo({Source.PassAccess()}, {Result.PassAccess()});");
    }

    public override void AppendGradientOp(List<Operation> ops, LayerRegistry registry, OperationFactory factory)
    {
        var sourceGradient = registry.GetOrCreateGradient(Source);
        var resultGradient = registry.GetGradient(Result);
        ops.Add(new ActivationGradientOperation(Source, sourceGradient));
        ops.Add(new PointwiseMultiplyOperation(sourceGradient, resultGradient, sourceGradient));
    }
}