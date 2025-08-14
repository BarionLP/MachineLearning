using System.Collections.Generic;

namespace ML.Analyzer.LayerFile.Operations;

internal sealed class VectorSingleMultiplyOperation(Weights vector, Weights single, Weights result, bool? add = null) : Operation
{
    public Weights Vector { get; } = vector.Type is NumberType.Vector ? vector : throw new InvalidOperationException();
    public Weights Single { get; } = single.Type is NumberType.Single ? single : throw new InvalidOperationException();
    public override Weights Result { get; } = result.Dimensions.SequenceEqual(vector.Dimensions) ? result : throw new InvalidOperationException();
    public bool Add { get; } = add ?? result.Location is Location.Gradients;

    public override void AppendCode(StringBuilder sb)
    {
        sb.AppendLine($$"""{{Vector.PassAccess()}}.Multiply{{(Add ? "Add" : "")}}To({{Single.PassAccess()}}, {{Result.PassAccess()}});""");
    }

    public override void AppendGradientOp(List<Operation> ops, LayerRegistry registry)
    {
        var resultGradient = registry.GetGradient(Result);
        ops.Add(new VectorSingleMultiplyOperation(resultGradient, Single, registry.GetOrCreateGradient(Vector)));
        ops.Add(new DotProductOperation(resultGradient, Vector, registry.GetOrCreateGradient(Single)));
    }
}
