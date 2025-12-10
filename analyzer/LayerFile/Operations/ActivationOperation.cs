using System.Collections.Generic;

namespace ML.Analyzer.LayerFile.Operations;

internal sealed class ActivationOperation(Weights source, Weights result, string location) : Operation
{
    public Weights Source { get; } = source;
    public override Weights Result { get; } = result;
    public string Location { get; } = location;

    public override void AppendCode(MethodBodyWriter sb)
    {
        sb.WriteOperation($"{Location}.ActivateTo({Source.PassAccess()}, {Result.PassAccess()});");
    }

    public override void AppendGradientOp(List<Operation> ops, LayerRegistry registry, OperationFactory factory)
    {
        var sourceGradient = registry.GetOrCreateGradient(Source);
        var resultGradient = registry.GetGradient(Result);
        ops.Add(new ActivationGradientOperation(Source, sourceGradient, Location));
        ops.Add(factory.NewPointwiseMultiply(sourceGradient, resultGradient, sourceGradient));
    }
}