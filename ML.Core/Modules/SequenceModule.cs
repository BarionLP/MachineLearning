using System.Runtime.InteropServices;
using Ametrin.Serializer;
using ML.Core.Modules.Initialization;

namespace ML.Core.Modules;

public sealed class SequenceModule<TArch> : IHiddenModule<TArch, SequenceModule<TArch>.Snapshot, SequenceModule<TArch>.Gradients>
{
    public required ImmutableArray<IHiddenModule<TArch>> Inner { get; init; }

    public TArch Forward(TArch input, Snapshot snapshot)
    {
        Debug.Assert(Inner.Length == snapshot.Inner.Length);
        return Inner.Zip(snapshot.Inner).Aggregate(input, static (input, m) => m.First.Forward(input, m.Second));
    }

    public TArch Backward(TArch outputGradient, Snapshot snapshot, Gradients gradients)
    {
        Debug.Assert(Inner.Length == snapshot.Inner.Length);
        Debug.Assert(Inner.Length == gradients.Inner.Length);

        foreach (var i in Inner.IndexRange.Reversed())
        {
            outputGradient = Inner[i].Backward(outputGradient, snapshot.Inner[i], gradients.Inner[i]);
        }

        return outputGradient;
    }

    public ulong ParameterCount => Inner.Sum(static m => m.ParameterCount);
    public Snapshot CreateSnapshot() => new(this);
    public Gradients CreateGradients() => new(this);

    public sealed class Snapshot(SequenceModule<TArch> module) : IModuleSnapshot
    {
        public ImmutableArray<IModuleSnapshot> Inner { get; } = [.. module.Inner.Select(static m => m.CreateSnapshot())];

        public void Dispose()
        {
            Inner.ForEach(static i => i.Dispose());
        }
    }

    public sealed class Gradients(SequenceModule<TArch> module) : IModuleGradients<Gradients>
    {
        public ImmutableArray<IModuleGradients> Inner { get; } = [.. module.Inner.Select(static m => m.CreateGradients())];

        public void Add(Gradients other)
        {
            Debug.Assert(Inner.Length == other.Inner.Length);
            foreach (var (left, right) in Inner.Zip(other.Inner))
            {
                left.Add(right);
            }
        }

        public void Reset()
        {
            Inner.ForEach(static m => m.Reset());
        }
    }

    static SequenceModule()
    {
        Training.AdamOptimizer.Registry.Register<SequenceModule<TArch>>(static (o, module) => new Adam(o, module));
    }

    public sealed class Adam(Training.AdamOptimizer optimizer, SequenceModule<TArch> module) : Training.IModuleOptimizer<Gradients>
    {
        public ImmutableArray<Training.IModuleOptimizer> SubOptimizers { get; } = [.. module.Inner.Select(optimizer.CreateModuleOptimizer)];
        public Training.AdamOptimizer Optimizer { get; } = optimizer;

        public void Apply(Gradients gradients)
        {
            Debug.Assert(gradients.Inner.Length == SubOptimizers.Length);
            SubOptimizers.Zip(gradients.Inner).ForEach(static p => p.First.Apply(p.Second));
        }

        public void FullReset()
        {
            SubOptimizers.ForEach(static sub => sub.FullReset());
        }
    }

    public sealed class SharedInitializer : IModuleInitializer<SequenceModule<TArch>>
    {
        public IModuleInitializer Inner { get; init; } = EmptyModuleInitializer.Instance;

        public SequenceModule<TArch> Init(SequenceModule<TArch> module)
        {
            module.Inner.ForEach(m => Inner.Init(m));
            return module;
        }
    }

    public sealed class Initializer : IModuleInitializer<SequenceModule<TArch>>
    {
        public required ImmutableArray<IModuleInitializer> Inner { get; init; }

        public SequenceModule<TArch> Init(SequenceModule<TArch> module)
        {
            module.Inner.Zip(Inner).ForEach(static p => p.Second.Init(p.First));
            return module;
        }
    }
}

public sealed class SequenceModuleConverter<TArch> : ISerializationConverter<SequenceModule<TArch>>
{
    public static Result<SequenceModule<TArch>, DeserializationError> TryReadValue(IAmetrinReader reader)
    {
        var modules = reader.TryReadArrayValue(AmetrinSerializer.TryReadDynamic<IHiddenModule<TArch>>);
        return modules.Map(static modules => new SequenceModule<TArch> { Inner = ImmutableCollectionsMarshal.AsImmutableArray(modules) });
    }

    public static void WriteValue(IAmetrinWriter writer, SequenceModule<TArch> value)
    {
        writer.WriteArrayValue(value.Inner.AsSpan(), AmetrinSerializer.WriteDynamic);
    }
}