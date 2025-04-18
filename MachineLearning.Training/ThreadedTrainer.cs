using System.Collections.Concurrent;
using System.Collections.Immutable;
using MachineLearning.Data.Entry;
using MachineLearning.Model.Layer.Snapshot;

namespace MachineLearning.Training;

public sealed class ThreadedTrainer
{
    public static TrainingContext Train(IEnumerable<TrainingData> trainingSet, ModelCachePool contextPool, ThreadingMode threading, Action<TrainingData, TrainingContext> action)
    {
        using var contexts = new ThreadLocal<TrainingContext>(() => new() { Gradients = contextPool.RentGradients() }, trackAllValues: true);
        var options = new ParallelOptions
        {
            MaxDegreeOfParallelism = threading switch
            {
                ThreadingMode.Single => 1,
                ThreadingMode.Half => Environment.ProcessorCount / 2,
                ThreadingMode.AlmostFull => Environment.ProcessorCount > 1 ? Environment.ProcessorCount - 1 : 1,
                ThreadingMode.Full => Environment.ProcessorCount, // setting MaxDegreeOfParallelism explicitly prevents too many presceduled tasks
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
            contextPool.Return(other.Gradients);
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
