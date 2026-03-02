using System.Collections.Generic;

namespace ML.Analyzer.LayerFile.Operations;

internal sealed class NestedLayerOperation(Module module, Weights input, Weights result) : Operation
{
    public Module Module { get; } = module;
    public Weights Input { get; } = input;
    public override Weights Result { get; } = result;

    public override void AppendCode(MethodBodyWriter sb)
    {
        sb.WriteOperation($"{Result.PassAccess()} = {Module.Access(Location.Pass)}.Forward({Input.PassAccess()}, {Module.AccessSnapshot(Location.Pass)});");
    }

    public override void AppendGradientOp(List<Operation> ops, LayerRegistry registry, OperationFactory factory)
    {
        ops.Add(new NestedLayerBackwardOperation(Module, Result, registry.CreateWeightsGradient(Input, preAllocate: false)));
    }
}

internal sealed class NestedLayerBackwardOperation(Module module, Weights input, Weights result) : Operation
{
    public Module Module { get; } = module;
    public Weights Input { get; } = input;
    public override Weights Result { get; } = result;

    public override void AppendCode(MethodBodyWriter sb)
    {
        sb.WriteOperation($"{Result.PassAccess()} = {Module.Access(Location.Pass)}.Backward({Input.PassAccess()}, {Module.AccessSnapshot(Location.Pass)}, {Module.AccessGradients(Location.Pass)});");
    }

    public override void AppendGradientOp(List<Operation> ops, LayerRegistry registry, OperationFactory factory)
    {
        throw new NotImplementedException();
    }
}
