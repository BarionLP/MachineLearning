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
}