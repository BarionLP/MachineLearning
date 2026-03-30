using ML.Core.Modules.Activations;
using ML.Core.Modules.Initialization;

namespace ML.Core.Modules.Builder;

public sealed class MultiLayerPerceptronBuilder
{
    private readonly List<(int input, int output, IActivationModule<Vector> activation)> layers = [];
    private int nextInput;
    public static MultiLayerPerceptronBuilder Create(int inputNodes) => new() { nextInput = inputNodes };

    public MultiLayerPerceptronBuilder AddLayer(int outputNodes, IActivationModule<Vector> activation)
    {
        layers.Add((nextInput, outputNodes, activation));
        nextInput = outputNodes;
        return this;
    }

    public MultiLayerPerceptronBuilder AddLayer(int outputNodes, Func<int, int, IActivationModule<Vector>> activation)
    {
        layers.Add((nextInput, outputNodes, activation.Invoke(nextInput, outputNodes)));
        nextInput = outputNodes;
        return this;
    }

    public SequenceModule<Vector> Build() => new()
    {
        Inner = [.. layers.SelectMany(d => (IEnumerable<IHiddenModule<Vector>>)[new LinearVectorModule(d.input, d.output), d.activation])],
    };

    public SequenceModule<Vector> BuildAndInit(Random random)
    {
        var module = Build();

        new SequenceModule<Vector>.Initializer
        {
            Inner = [.. GetIniters()],
        }.Init(module);

        return module;

        IEnumerable<IModuleInitializer> GetIniters()
        {
            foreach (var i in module.Inner.IndexRange)
            {
                var subModule = module.Inner[i];
                var nextSubModule = i < (module.Inner.Length - 1) ? module.Inner[i + 1] : null;

                yield return (subModule, nextSubModule) switch
                {
                    (IActivationModule, _) => EmptyModuleInitializer.Instance,
                    (LinearVectorModule, SoftMaxActivation) => new LinearVectorModule.XavierInitializer() { Random = random },
                    (LinearVectorModule, LeakyReLUActivation) => new LinearVectorModule.KaimingInitializer((IActivationModule)nextSubModule) { Random = random },
                    _ => throw new NotImplementedException(),
                };
            }
        }
    }
}