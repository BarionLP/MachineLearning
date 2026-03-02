using ML.Core.Attributes;

namespace ML.Core.Modules;

[GeneratedModule]
public sealed partial class EmbeddedModule<TIn, TArch, TOut> : IModule<TArch>
{
    [SubModule] public required IInputModule<TIn, TArch> Input { get; init; }
    [SubModule] public required IHiddenModule<TArch> Hidden { get; init; }
    [SubModule] public required IOutputModule<TArch, TOut> Output { get; init; }

    public (TOut Output, Weight Confidence, TArch Weights) Forward(TIn input, Snapshot snapshot)
    {
        return Output.Forward(Hidden.Forward(Input.Forward(input, snapshot.Input), snapshot.Hidden), snapshot.Output);
    }

    public TArch Backward(TArch outputGradient, Snapshot snapshot, Gradients gradients)
    {
        return Input.Backward(Hidden.Backward(Output.Backward(outputGradient, snapshot.Output, gradients.Output), snapshot.Hidden, gradients.Hidden), snapshot.Input, gradients.Input);
    }
}
