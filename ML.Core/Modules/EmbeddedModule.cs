using ML.Core.Attributes;
using ML.Core.Modules.Initialization;
using ML.Core.Training;

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

    static EmbeddedModule()
    {
        AdamOptimizer.Registry.Register<EmbeddedModule<TIn, TArch, TOut>>(static (o, module) => new Adam(o, module));
    }

    public sealed class Adam(AdamOptimizer optimizer, EmbeddedModule<TIn, TArch, TOut> module) : IModuleOptimizer<Gradients>
    {
        public IModuleOptimizer Input { get; } = optimizer.CreateModuleOptimizer(module.Input);
        public IModuleOptimizer Hidden { get; } = optimizer.CreateModuleOptimizer(module.Hidden);
        public IModuleOptimizer Output { get; } = optimizer.CreateModuleOptimizer(module.Output);

        public void Apply(Gradients gradients)
        {
            Input.Apply(gradients.Input);
            Hidden.Apply(gradients.Hidden);
            Output.Apply(gradients.Output);
        }

        public void FullReset()
        {
            Input.FullReset();
            Hidden.FullReset();
            Output.FullReset();
        }
    }

    public sealed class Initializer : IModuleInitializer<EmbeddedModule<TIn, TArch, TOut>>
    {
        public IModuleInitializer Input { get; init; } = EmptyModuleInitializer.Instance;
        public IModuleInitializer Hidden { get; init; } = EmptyModuleInitializer.Instance;
        public IModuleInitializer Output { get; init; } = EmptyModuleInitializer.Instance;

        public void Init(EmbeddedModule<TIn, TArch, TOut> module)
        {
            Input.Init(module.Input);
            Hidden.Init(module.Hidden);
            Output.Init(module.Output);
        }
    }
}
