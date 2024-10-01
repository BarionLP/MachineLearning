using MachineLearning.Training.Evaluation;
using MachineLearning.Training.Optimization;

namespace MachineLearning.Training;

public sealed class LegacyModelTrainer<TInput, TOutput> where TInput : notnull where TOutput : notnull
{
    public TrainingConfig<TInput, TOutput> Config { get; }
    public EmbeddedModel<TInput, TOutput> Model { get; }
    public IGenericOptimizer Optimizer { get; }
    internal LegacyTrainingContext<TInput, TOutput> Context { get; }

    public LegacyModelTrainer(EmbeddedModel<TInput, TOutput> model, TrainingConfig<TInput, TOutput> config)
    {
        Config = config;
        Model = model;
        Optimizer = config.Optimizer;
        Context = new(model, config, Optimizer);
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

        Console.WriteLine($"Training {Model}");
        Console.WriteLine(Config.ToString());
        Console.WriteLine("Starting Training...");
        Train(cts.Token);
        cts.Cancel();
        Console.WriteLine("Training Done!");
    }

    public void Train(CancellationToken? token = null)
    {
        //var before = EvaluateShort();
        Optimizer.Init();
        Context.FullReset();
        var cachedEvaluation = DataSetEvaluationResult.ZERO;
        foreach (var epochIndex in ..Config.EpochCount)
        {
            var epoch = Config.GetEpoch();
            var batchCount = 0;

            foreach (var batch in epoch)
            {
                cachedEvaluation += Context.TrainAndEvaluate(batch, multithread: true);
                if ((Config.DumpBatchEvaluation && batchCount % Config.DumpEvaluationAfterBatches == 0) || (batchCount + 1 == epoch.BatchCount && Config.DumpEpochEvaluation))
                {
                    Config.EvaluationCallback!.Invoke(new DataSetEvaluation { Context = GetContext(), Result = cachedEvaluation });
                    cachedEvaluation = DataSetEvaluationResult.ZERO;
                }
                batchCount++;
                Optimizer.OnBatchCompleted();

                if (token?.IsCancellationRequested is true)
                {
                    Optimizer.OnEpochCompleted();
                    return;
                }
            }

            Optimizer.OnEpochCompleted();

            TrainingEvaluationContext GetContext() => new()
            {
                CurrentBatch = batchCount + 1,
                MaxBatch = epoch.BatchCount,
                CurrentEpoch = epochIndex + 1,
                MaxEpoch = Config.EpochCount,
                LearnRate = Config.Optimizer.LearningRate,
            };
        }
    }

    public ModelEvaluationResult EvaluateShort() => new()
    {
        TrainingSetResult = Evaluator.Evaluate(Model, Config.Optimizer.CostFunction, Config.OutputResolver, Config.GetRandomTrainingBatch()),
        TestSetResult = Evaluator.Evaluate(Model, Config.Optimizer.CostFunction, Config.OutputResolver, Config.GetRandomTestBatch()),
    };
}
