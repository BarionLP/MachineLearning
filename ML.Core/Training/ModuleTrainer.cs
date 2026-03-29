using System.Buffers;
using System.Text;
using System.Threading;
using ML.Core.Evaluation;
using ML.Core.Evaluation.Cost;
using ML.Core.Modules;
using ML.Core.Data.Training;

namespace ML.Core.Training;

public sealed class EmbeddedModuleTrainer<TIn, TArch, TOut>
    where TArch : ITensorLike<TArch>
{
    public EmbeddedModule<TIn, TArch, TOut> Module { get; }
    public TrainingConfig Config { get; }
    public required ITrainingDataSource<TrainingEntry<TIn, TArch, TOut>> TrainingData { get; init; }
    public required ICostFunction<TArch> CostFunction { get; init; }
    public ModuleDataPool DataPool { get; }

    private Optimizer Optimizer => Config.Optimizer;
    private readonly IModuleOptimizer moduleOptimizer;

    public EmbeddedModuleTrainer(EmbeddedModule<TIn, TArch, TOut> module, TrainingConfig config)
    {
        Module = module;
        Config = config;
        moduleOptimizer = Optimizer.CreateModuleOptimizer(module);
        DataPool = new(module);
    }

    public void TrainConsole(bool cancelable = true)
    {
        using var cts = new CancellationTokenSource();
        if (cancelable)
        {
            Task.Run(async () =>
            {
                while (!cts.IsCancellationRequested)
                {
                    if (Console.KeyAvailable && Console.ReadKey(intercept: true).Key == ConsoleKey.C)
                    {
                        Console.WriteLine("Canceling...");
                        cts.Cancel();
                        break;
                    }
                    await Task.Delay(500);
                }
            });
        }

        Console.WriteLine($"Training {Module} ({Module.ParameterCount})");
        Console.WriteLine(GenerateTrainingOverview(Config, TrainingData.BatchCount, TrainingData.BatchSize));
        Console.WriteLine("Starting Training...");
        Console.WriteLine(TrainingEvaluationResult.GetHeader());
        Train(cts.Token);
        cts.Cancel();
        Console.WriteLine("Training Done!");
    }


    public void Train(CancellationToken token = default)
    {
        Optimizer.Init();
        var runningEvaluation = EvaluationResult.ZERO;

        foreach (var epochIndex in ..Config.EpochCount)
        {
            TrainingData.Reset();

            foreach (var (batchIndex, batch) in TrainingData.GetBatches().Index())
            {
                runningEvaluation += RunBatch(batch);

                if ((Config.BatchEvaluationEnabled && batchIndex % Config.EvaluationCallbackAfterBatches is 0)
                    || (batchIndex + 1 == TrainingData.BatchCount && Config.EpochEvaluationEnabled))
                {
                    Config.EvaluationCallback!.Invoke(new TrainingEvaluationResult { Context = GetContext(), Result = runningEvaluation });
                    runningEvaluation = EvaluationResult.ZERO;
                }

                Optimizer.OnBatchCompleted();

                if (token.IsCancellationRequested)
                {
                    Optimizer.OnEpochCompleted();
                    return;
                }

                TrainingEvaluationContext GetContext() => new()
                {
                    CurrentBatch = batchIndex + 1,
                    MaxBatch = TrainingData.BatchCount,
                    CurrentEpoch = epochIndex + 1,
                    MaxEpoch = Config.EpochCount,
                    LearningRate = Optimizer.LearningRate,
                };
            }

            Optimizer.OnEpochCompleted();
        }
    }

    public EvaluationResult RunBatch(IEnumerable<TrainingEntry<TIn, TArch, TOut>> batch)
    {
        var timeStamp = Stopwatch.GetTimestamp();

        using var context = ThreadedTrainer.Train(batch, DataPool, Config.Threading, (entry, context) =>
        {
            var (output, condfidence, cost) = RunEntry(entry, (EmbeddedModule<TIn, TArch, TOut>.Gradients)context.Gradients);
            if (EqualityComparer<TOut>.Default.Equals(output, entry.ExpectedValue))
            {
                context.CorrectCount++;
                context.CorrectConfidenceSum += condfidence;
            }
            else
            {
                context.WrongConfidenceSum += condfidence;
            }

            context.TotalCount++;
            context.TotalCost += cost;
        });


        moduleOptimizer.Apply(context.Gradients);

        return new()
        {
            TotalCount = context.TotalCount,
            CorrectCount = context.CorrectCount,
            CorrectConfidenceSum = context.CorrectConfidenceSum,
            WrongConfidenceSum = context.WrongConfidenceSum,
            TotalCost = context.TotalCost,
            TotalElapsedTime = Stopwatch.GetElapsedTime(timeStamp),
        };
    }

    private (TOut output, Weight confidence, Weight cost) RunEntry(TrainingEntry<TIn, TArch, TOut> entry, EmbeddedModule<TIn, TArch, TOut>.Gradients gradients)
    {
        using var marker = DataPool.RentSnapshot();
        var snapshot = (EmbeddedModule<TIn, TArch, TOut>.Snapshot)marker.Snapshot;

        var (output, condfidence, outputWeights) = Module.Forward(entry.InputValue, snapshot);

        NumericsDebug.AssertSameDimensions(outputWeights, entry.ExpectedWeights);
        using var outputGradientStorage = ArrayPool<Weight>.Shared.RentNumerics(outputWeights.FlatCount);
        var outputGradient = TArch.OfSize(outputWeights, outputGradientStorage);
        CostFunction.DerivativeTo(outputWeights, entry.ExpectedWeights, outputGradient);

        var inputGradient = Module.Backward(outputGradient, snapshot, gradients);

        return (output, condfidence, CostFunction.TotalCost(outputWeights, entry.ExpectedWeights));
    }

    public void FullReset()
    {
        moduleOptimizer.FullReset();
    }

    public static string GenerateTrainingOverview(TrainingConfig config, int batchCount, int batchSize)
    {
        var sb = new StringBuilder();
        sb.AppendLine();
        sb.AppendLine("Training Info:");
        sb.AppendLine($"using {config.Optimizer.GetType().Name} ({config.Threading})");
        sb.AppendLine("Training for");
        sb.AppendLine($" - {config.EpochCount} epochs");
        sb.AppendLine($"  - {batchCount} batches");
        sb.AppendLine($"   - {batchSize} entries");

        if (config.EvaluationCallbackEnabled)
        {
            if (config.BatchEvaluationEnabled)
            {
                if (config.EvaluationCallbackAfterBatches == 1)
                {
                    sb.AppendLine("Dumping every batch");
                }
                else
                {
                    sb.AppendLine($"Dumping every {config.EvaluationCallbackAfterBatches} batches");
                }
            }
            else
            {
                sb.AppendLine($"Dumping every epoch");
            }
        }

        sb.AppendLine();
        return sb.ToString();
    }
}