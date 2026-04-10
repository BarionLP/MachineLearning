using ML.Core.Attributes;

namespace ML.Core.Modules.Activations;

/// <summary>
/// Gaussian Error Linear Unit
/// </summary>
[GeneratedModule(IncludeSerializer: true)]
public sealed partial class GeLUActivation : IActivationModule<Vector, SimpleModuleSnapshot>
{
    public static GeLUActivation Instance => field ??= new();

    public Vector Forward(Vector input, SimpleModuleSnapshot snapshot)
    {
        snapshot.Input = input;
        SpanOperations.MapTo<GeLUOp, Empty>(default, input.AsSpan(), snapshot.Output.AsSpan());
        return snapshot.Output;
    }

    public Vector Backward(Vector outputGradient, SimpleModuleSnapshot snapshot, EmptyModuleData gradients)
    {
        SpanOperations.MapTo<GeLUGradientOp, Empty>(default, snapshot.Input.AsSpan(), outputGradient.AsSpan(), snapshot.InputGradient.AsSpan());
        return snapshot.InputGradient;
    }

    private const Weight Coeff = (Weight)0.7978845608028654; // Weight.Sqrt(2 / Weight.Pi);
    private const Weight Alpha = (Weight)0.044715;
    private readonly ref struct GeLUOp : IUnaryOperator<Empty>
    {
        public static Weight Invoke(in Empty _, Weight input)
        {
            var input3 = input * input * input;
            var inner = Coeff * (input + Alpha * input3);
            return (Weight)0.5 * input * (1 + Weight.Tanh(inner));
        }

        public static SimdVector Invoke(in Empty _, SimdVector input)
        {
            var input3 = input * input * input;
            var inner = Coeff * (input + Alpha * input3);
            return (Weight)0.5 * input * (SimdVectorHelper.Create<Weight>(1) + TanhOperator.Invoke(in _, inner));
        }
    }

    private readonly ref struct GeLUGradientOp : IBinaryOperator<Empty>
    {
        private const Weight Alpha3 = (Weight)0.134145; // 3 * Alpha

        public static Weight Invoke(in Empty _, Weight input, Weight gradient)
        {
            var input2 = input * input;
            var input3 = input2 * input;

            var inner = Coeff * (input + Alpha * input3);
            var t = Weight.Tanh(inner);
            var sech2 = 1 - t * t;
            var dGeLU = (Weight)0.5 * (1 + t) + (Weight)0.5 * input * sech2 * Coeff * (1 + Alpha3 * input2);
            return dGeLU * gradient;
        }

        public static SimdVector Invoke(in Empty _, SimdVector input, SimdVector gradient)
        {
            var input2 = input * input;
            var input3 = input2 * input;

            var inner = Coeff * (input + Alpha * input3);
            var t = TanhOperator.Invoke(in _, inner);
            var sech2 = SimdVector.One - t * t;
            var dGeLU = (Weight)0.5 * (SimdVector.One + t) + (Weight)0.5 * input * sech2 * Coeff * (SimdVector.One + Alpha3 * input2);
            return dGeLU * gradient;
        }
    }
}