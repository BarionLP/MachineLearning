using System.Runtime.InteropServices;

namespace MachineLearning.Domain.Activation;

public sealed class SoftmaxActivation : IActivationMethod
{
    public static readonly SoftmaxActivation Instance = new();

    public void Activate(Vector input, Vector result)
    {
        input.Map(Math.Exp, result);
        var sum = result.Sum();
        result.DivideInPlace(sum);
    }

    public void Derivative(Vector input, Vector result)
    {
        input.Map(Math.Exp, result);
        var sum = result.Sum();
        var inverseSumSquared = 1 / (sum * sum);
        //result.MapInPlace(ex => (ex * sum - ex * ex) * inverseSumSquared);

        ref var vectorPtr = ref MemoryMarshal.GetReference(result.AsSpan());
        ref var resultPtr = ref MemoryMarshal.GetReference(result.AsSpan());
        var mdSize = (nuint) SimdVector.Count;
        var length = (nuint) result.Count;

        nuint index = 0;
        for(; index + mdSize <= length; index += mdSize)
        {
            var simdVector = SimdVectorHelper.LoadUnsafe(ref vectorPtr, index);
            SimdVectorHelper.StoreUnsafe((simdVector * sum - simdVector * simdVector) * inverseSumSquared, ref resultPtr, index);
        }

        for(; index < length; index++)
        {
            result[index] = (result[index] * sum - result[index] * result[index]) * inverseSumSquared;
        }
    }
}
