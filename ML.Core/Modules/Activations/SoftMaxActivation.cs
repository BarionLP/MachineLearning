using System.Runtime.InteropServices;
using ML.Core.Attributes;

namespace ML.Core.Modules.Activations;

[GeneratedModule]
public sealed partial class SoftMaxActivation(int inputNodes) : IHiddenModule<Vector, SoftMaxActivation.Snapshot, EmptyModuleData>, IActivationModule
{
    [Property] public int InputNodes { get; } = inputNodes;

    public Vector Forward(Vector input, Snapshot snapshot)
    {
        Debug.Assert(input.Count == InputNodes);
        snapshot.Input = input;
        snapshot.Input.SoftMaxTo(snapshot.Output);
        return snapshot.Output;
    }

    public Vector Backward(Vector outputGradient, Snapshot snapshot, EmptyModuleData gradients)
    {
        var input = snapshot.Input;
        var result = snapshot.OutputGradient;
        var max = input.Max();
        input.SubtractPointwiseTo(max, result);
        result.PointwiseExpToSelf();
        var sum = result.Sum();
        var inverseSumSquared = 1 / (sum * sum);

        ref var vectorPtr = ref MemoryMarshal.GetReference(result.AsSpan());
        ref var resultPtr = ref MemoryMarshal.GetReference(result.AsSpan());
        var mdSize = (nuint)SimdVector.Count;
        var length = (nuint)result.Count;

        nuint index = 0;
        for (; index + mdSize <= length; index += mdSize)
        {
            var simdVector = SimdVectorHelper.LoadUnsafe(ref vectorPtr, index);
            SimdVectorHelper.StoreUnsafe((simdVector * sum - simdVector * simdVector) * inverseSumSquared, ref resultPtr, index);
        }

        for (; index < length; index++)
        {
            var value = result[index];
            result[index] = (value * sum - value * value) * inverseSumSquared;
        }

        NumericsDebug.AssertValidNumbers(result);

        return result;
    }

    public sealed class Snapshot(SoftMaxActivation module) : IModuleSnapshot
    {
        public Vector Input { get; set; }
        public Vector Output { get; } = Vector.Create(module.InputNodes);
        public Vector OutputGradient { get; } = Vector.Create(module.InputNodes);
    }
}