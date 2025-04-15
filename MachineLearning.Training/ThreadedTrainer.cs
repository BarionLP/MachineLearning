using System.Collections.Concurrent;
using System.Collections.Immutable;
using MachineLearning.Data.Entry;
using MachineLearning.Model.Layer.Snapshot;

namespace MachineLearning.Training;

public sealed class ThreadedTrainer
{
    public static TrainingContext Train(IEnumerable<TrainingData> trainingSet, TrainingContextPool contextPool, ThreadingMode threading, Action<TrainingData, TrainingContext> action)
    {
        using var contexts = new ThreadLocal<TrainingContext>(contextPool.Rent, trackAllValues: true);
        var options = new ParallelOptions
        {
            MaxDegreeOfParallelism = threading switch
            {
                ThreadingMode.Single => 1,
                ThreadingMode.Half => Environment.ProcessorCount / 2,
                ThreadingMode.AlmostFull => Environment.ProcessorCount > 1 ? Environment.ProcessorCount - 1 : 1,
                ThreadingMode.Full => -1,
                _ => throw new UnreachableException()
            },
        };
        var partitioner = Partitioner.Create(trainingSet);
        var result = Parallel.ForEach(partitioner, options, (item, state) =>
        {
            action(item, contexts.Value!);
        });

        Debug.Assert(result.IsCompleted);

        var context = contexts.Values[0];

        foreach (var other in contexts.Values.Skip(1))
        {
            context.Add(other);
            contextPool.Return(other);
        }

        return context;
    }
}

public enum ThreadingMode { Single, Half, Full, AlmostFull }

public sealed class TrainingContext
{
    public int TotalCount { get; set; }
    public int CorrectCount { get; set; }
    public float TotalCost { get; set; }
    public required ImmutableArray<IGradients> Gradients { get; init; }

    public void Add(TrainingContext other)
    {
        TotalCount += other.TotalCount;
        CorrectCount += other.CorrectCount;
        TotalCost += other.TotalCost;

        foreach (var (g, o) in Gradients.Zip(other.Gradients))
        {
            g.Add(o);
        }
    }

    public void Reset()
    {
        TotalCount = 0;
        CorrectCount = 0;
        TotalCost = 0;

        foreach (var gradient in Gradients)
        {
            gradient.Reset();
        }
    }
}

public sealed class TrainingContextPool(Func<ImmutableArray<IGradients>> gradientGetter)
{
    private readonly ConcurrentBag<TrainingContext> cache = [];
    public int UnusedItems => cache.Count;

    public TrainingContext Rent()
    {
        if (cache.TryTake(out var context))
        {
            return context;
        }

        return new TrainingContext { Gradients = gradientGetter() };
    }

    public void Return(TrainingContext context)
    {
        context.Reset();
        cache.Add(context);
    }

    public void Clear()
    {
        cache.Clear();
    }
}