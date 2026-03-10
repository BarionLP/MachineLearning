using ML.Core.Attributes;

namespace ML.Core.Modules.Activations;

[GeneratedModule]
public sealed partial class LeakyReLUActivation(Weight alpha = 0.01f) : IActivationModule<Vector, LeakyReLUActivation.Snapshot>
{
    public static LeakyReLUActivation Instance => field ??= new();

    [Property] public Weight Alpha { get; } = alpha;
    private readonly LeakyReLUOperation forwardOp = new(alpha);
    private readonly LeakyReLUDerivativeOperation derivativeOp = new(alpha);

    public Vector Forward(Vector input, Snapshot snapshot)
    {
        snapshot.Input = input;
        snapshot.Input.MapTo(forwardOp, snapshot.Output);
        return snapshot.Output;
    }

    public Vector Backward(Vector outputGradient, Snapshot snapshot, EmptyModuleData gradients)
    {
        snapshot.Input.MapTo(derivativeOp, snapshot.InputGradient);
        snapshot.InputGradient.PointwiseMultiplyToSelf(outputGradient);
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
        public Vector Output => outputHandle.Vector;
        public Vector InputGradient => inputGradientHandle.Vector;

        private DynamicVector outputHandle = new();
        private DynamicVector inputGradientHandle = new();

        internal Snapshot(LeakyReLUActivation _) : this() { }

        public void Dispose()
        {
            outputHandle.Dispose();
            inputGradientHandle.Dispose();
        }
    }

    public readonly struct LeakyReLUOperation(Weight alpha) : IUnaryOperator<LeakyReLUOperation>
    {
        private readonly Weight alpha = alpha;
        // constructing an alpha vector once and reusing seems to be slower

        public static Weight Invoke(in LeakyReLUOperation info, Weight input) => input > 0 ? input : info.alpha * input;
        public static SimdVector Invoke(in LeakyReLUOperation info, SimdVector input)
            => SimdVectorHelper.ConditionalSelect(SimdVectorHelper.GreaterThan(input, SimdVector.Zero), input, input * info.alpha);
    }

    private readonly struct LeakyReLUDerivativeOperation(Weight alpha) : IUnaryOperator<LeakyReLUDerivativeOperation>
    {
        private readonly Weight alpha = alpha;
        // constructing an alpha vector once and reusing seems to be slower

        public static Weight Invoke(in LeakyReLUDerivativeOperation info, Weight input) => input > 0 ? 1 : info.alpha;
        public static SimdVector Invoke(in LeakyReLUDerivativeOperation info, SimdVector input)
            => SimdVectorHelper.ConditionalSelect(SimdVectorHelper.GreaterThan(input, SimdVector.Zero), SimdVector.One, SimdVectorHelper.Create(info.alpha));
    }
}