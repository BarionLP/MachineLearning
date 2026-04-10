using ML.Core.Attributes;

namespace ML.Core.Modules.Activations;

[GeneratedModule(IncludeSerializer: true)]
public sealed partial class LeakyReLUActivation(Weight alpha = 0.01f) : IActivationModule<Vector, LeakyReLUActivation.Snapshot>, IActivationModule<Matrix, LeakyReLUActivation.Snapshot>
{
    public static LeakyReLUActivation Instance => field ??= new();

    public Weight Alpha { get; } = alpha;

    public Vector Forward(Vector input, Snapshot snapshot)
    {
        snapshot.Input = input;
        SpanOperations.MapTo<LeakyReLUOp, Weight>(Alpha, snapshot.Input.AsSpan(), snapshot.Output.AsSpan());
        return snapshot.Output;
    }

    public Vector Backward(Vector outputGradient, Snapshot snapshot, EmptyModuleData gradients)
    {
        SpanOperations.MapTo<LeakyReLUGradientOp, Weight>(Alpha, snapshot.Input.AsSpan(), outputGradient.AsSpan(), snapshot.InputGradient.AsSpan());
        return snapshot.InputGradient;
    }

    public Matrix Forward(Matrix input, Snapshot snapshot) => Matrix.OfSize(input, Forward(input.Storage, snapshot));
    public Matrix Backward(Matrix outputGradient, Snapshot snapshot, EmptyModuleData gradients) => Matrix.OfSize(outputGradient, Backward(outputGradient.Storage, snapshot, gradients));
    
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

        private readonly Dynamic<Vector> outputHandle = new();
        private readonly Dynamic<Vector> inputGradientHandle = new();

        internal Snapshot(LeakyReLUActivation _) : this() { }

        public void Dispose()
        {
            outputHandle.Dispose();
            inputGradientHandle.Dispose();
        }
    }

    private readonly ref struct LeakyReLUOp : IUnaryOperator<Weight>
    {
        public static Weight Invoke(in Weight alpha, Weight input) => input > 0 ? input : alpha * input;
        public static SimdVector Invoke(in Weight alpha, SimdVector input)
            => SimdVectorHelper.ConditionalSelect(SimdVectorHelper.GreaterThan(input, SimdVector.Zero), input, input * alpha);
    }

    private readonly ref struct LeakyReLUGradientOp : IBinaryOperator<Weight>
    {
        public static Weight Invoke(in Weight alpha, Weight input, Weight gradient) => input > 0 ? gradient : gradient * alpha;
        public static SimdVector Invoke(in Weight alpha, SimdVector input, SimdVector gradient)
            => SimdVectorHelper.ConditionalSelect(SimdVectorHelper.GreaterThan(input, SimdVector.Zero), gradient, gradient * alpha);
    }

    IModuleSnapshot IModule.CreateSnapshot() => CreateSnapshot();
    IModuleGradients IModule.CreateGradients() => CreateGradients();
}
