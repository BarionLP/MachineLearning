using System.Threading;
using ML.Core.Modules;

namespace ML.Core.Training;

public sealed class ThreadedTrainer
{
    public static TrainingContext Train<T>(IEnumerable<T> trainingSet, ModuleDataPool dataPool, ThreadingMode threading, Action<T, TrainingContext> action)
    {
        if (threading is ThreadingMode.Single)
        {
            var localContext = new TrainingContext { Pool = dataPool };

            foreach (var item in trainingSet)
            {
                action(item, localContext);
            }

            return localContext;
        }

        using var contexts = new ThreadLocal<TrainingContext>(() => new() { Pool = dataPool }, trackAllValues: true);
        var options = new ParallelOptions
        {
            MaxDegreeOfParallelism = threading switch
            {
                // ThreadingMode.Single => 1,
                ThreadingMode.Half => Environment.ProcessorCount / 2,
                ThreadingMode.AlmostFull => Environment.ProcessorCount > 1 ? Environment.ProcessorCount - 1 : 1,
                ThreadingMode.Full => Environment.ProcessorCount, // setting MaxDegreeOfParallelism explicitly prevents too many presceduled tasks
                _ => throw new UnreachableException()
            },
        };
        var result = Parallel.ForEach(trainingSet, options, (item, state) =>
        {
            action(item, contexts.Value!);
        });

        Debug.Assert(result.IsCompleted);

        var context = contexts.Values[0];

        foreach (var other in contexts.Values.Skip(1))
        {
            context.Add(other);
            other.Dispose();
        }

        return context;
    }
}

public sealed class TrainingContext : IDisposable
{
    public int TotalCount { get; set; }
    public int CorrectCount { get; set; }
    public float CorrectConfidenceSum { get; set; }
    public float WrongConfidenceSum { get; set; }
    public float TotalCost { get; set; }
    public required ModuleDataPool Pool { get; init; }
    private IModuleGradients? _gradients;
    public IModuleGradients Gradients => _gradients ??= Pool.RentGradients();

    public void Add(TrainingContext other)
    {
        TotalCount += other.TotalCount;
        CorrectCount += other.CorrectCount;
        CorrectConfidenceSum += other.CorrectConfidenceSum;
        WrongConfidenceSum += other.WrongConfidenceSum;
        TotalCost += other.TotalCost;

        Gradients.Add(other.Gradients);
    }

    public void Dispose()
    {
        TotalCount = 0;
        CorrectCount = 0;
        TotalCost = 0;

        if (_gradients is not null)
        {
            Pool.Return(_gradients);
        }
    }
}