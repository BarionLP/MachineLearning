using System.Collections.Concurrent;
using System.Collections.Immutable;
using MachineLearning.Data.Entry;
using MachineLearning.Model.Layer.Snapshot;

namespace MachineLearning.Training;

public sealed class ThreadedTrainer
{
    public static TrainingContext Train(IEnumerable<TrainingData> trainingSet, Func<ImmutableArray<IGradients>> gradientGetter, int threads, Action<TrainingData, TrainingContext> action)
    {
        var contexts = new ConcurrentQueue<TrainingContext>();
        var options = new ParallelOptions { MaxDegreeOfParallelism = threads };
        var result = Parallel.ForEach(trainingSet, options, () => new TrainingContext { Gradients = gradientGetter() }, (item, state, context) =>
        {
            action(item, context);
            return context;
        }, contexts.Enqueue);

        Debug.Assert(result.IsCompleted);
        Debug.Assert(!contexts.IsEmpty);

        if (contexts.TryDequeue(out var context))
        {
            while (contexts.TryDequeue(out var other))
            {
                context.Add(other);
            }
            return context;
        }

        return new() { Gradients = gradientGetter() };
    }
}

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
}
