using ML.Core.Attributes;

namespace ML.Core.Modules.Activations;

[GeneratedModule(IncludeSerializer: true)]
public sealed partial class SoftMaxActivation : IActivationModule<Vector, SoftMaxActivation.Snapshot>
{
    public static SoftMaxActivation Instance => field ??= new();
    public Vector Forward(Vector input, Snapshot snapshot)
    {
        snapshot.Input = input;
        snapshot.Input.SoftMaxTo(snapshot.Output);
        return snapshot.Output;
    }

    public Vector Backward(Vector outputGradient, Snapshot snapshot, EmptyModuleData gradients)
    {
        var dot = snapshot.Output.Dot(outputGradient);
        outputGradient.SubtractPointwiseTo(dot, snapshot.InputGradient);
        snapshot.InputGradient.PointwiseMultiplyToSelf(snapshot.Output);
        NumericsDebug.AssertValidNumbers(snapshot.InputGradient);
        return snapshot.InputGradient;
    }

    public sealed class Snapshot() : IModuleSnapshot
    {
        public Vector Input
        {
            get;
            set
            {
                field = value;
                outputHandle.SetCount(field.Count);
                inputGradientHandle.SetCount(field.Count);
            }
        }
        public Vector Output => outputHandle.Tensor;
        public Vector InputGradient => inputGradientHandle.Tensor;

        private Dynamic<Vector> outputHandle = new();
        private Dynamic<Vector> inputGradientHandle = new();

        internal Snapshot(SoftMaxActivation _) : this() { }

        public void Dispose()
        {
            outputHandle.Dispose();
            inputGradientHandle.Dispose();
        }
    }
}
