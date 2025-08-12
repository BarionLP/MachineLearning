using System.Collections.Generic;

namespace ML.Analyzer.LayerFile.Operations;

internal sealed class OutputOperation(Weights weights) : Operation
{
    public Weights Weights { get; } = weights;
    public override Weights Result => Weights;

    public override void AppendCode(StringBuilder sb)
    {
        sb.AppendLine($"return {Weights.PassAccess()};");
    }

    public override void AppendGradientOp(List<Operation> ops, LayerRegistry registry)
    {
        var input = new DirectWeights("outputGradient", Weights.Dimensions, Location.Pass); // defined as method parameter
        registry.AddAlias(((DirectWeights)Weights).GetGradientName(), input);
        // var result = registry.CreateGradient(Weights, Location.Pass);
        // ops.Add(new DefineOperation(input, result));
    }
}
