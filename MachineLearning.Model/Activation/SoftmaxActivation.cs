using System.Runtime.InteropServices;

namespace MachineLearning.Model.Activation;

public sealed class SoftMaxActivation(Weight temperature) : IActivationFunction
{
    public static readonly SoftMaxActivation Instance = new(1);

    public Weight Temperature { get; } = temperature;
    public void Activate(Vector input, Vector result) => input.Divide(1).SoftMax(result);
    public void Derivative(Vector input, Vector result)
    {
        input.PointwiseExp(result);
        var sum = result.Sum();
        var inverseSumSquared = 1 / (sum * sum);
        var inverseTemperature = 1; /// Temperature;

        ref var vectorPtr = ref MemoryMarshal.GetReference(result.AsSpan());
        ref var resultPtr = ref MemoryMarshal.GetReference(result.AsSpan());
        var mdSize = (nuint)SimdVector.Count;
        var length = (nuint)result.Count;

        nuint index = 0;
        for (; index + mdSize <= length; index += mdSize)
        {
            var simdVector = SimdVectorHelper.LoadUnsafe(ref vectorPtr, index);
            SimdVectorHelper.StoreUnsafe((simdVector * sum - simdVector * simdVector) * inverseSumSquared * inverseTemperature, ref resultPtr, index);
        }

        for (; index < length; index++)
        {
            result[index] = (result[index] * sum - result[index] * result[index]) * inverseSumSquared * inverseTemperature;
        }
    }
}
