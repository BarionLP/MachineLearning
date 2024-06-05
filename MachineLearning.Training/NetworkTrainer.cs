using MachineLearning.Model;
using MachineLearning.Model.Layer;
using MachineLearning.Training.Evaluation;
using MachineLearning.Training.Optimization;
using System.Diagnostics;

namespace MachineLearning.Training;

public sealed class NetworkTrainer<TInput, TOutput> where TInput : notnull where TOutput : notnull
{
    public TrainingConfig<TInput, TOutput> Config { get; }
    public INetwork<TInput, TOutput, RecordingLayer> Network { get; }
    public IOptimizer Optimizer { get; }
    internal ModelTrainingContext<TInput, TOutput> Context { get; }

    public NetworkTrainer(TrainingConfig<TInput, TOutput> config, INetwork<TInput, TOutput, RecordingLayer> network)
    {
        Config = config;
        Network = network;
        Optimizer = config.Optimizer.CreateOptimizer();
        Context = new(network, config, Optimizer);
    }

    public ModelTrainingResult Train()
    {
        // for each epoch 
        // train on all batches

        //var before = EvaluateShort();
        Optimizer.Init();
        Context.FullReset();
        var sw = Stopwatch.StartNew();
        foreach(var epochIndex in ..Config.EpochCount)
        {
            var epoch = Config.GetEpoch();
            var batchCount = 0;

            foreach(var batch in epoch)
            {
                var evaluation = Context.TrainAndEvaluate(batch);
                if((Config.DumpBatchEvaluation && batchCount % Config.DumpEvaluationAfterBatches == 0) || (batchCount + 1 == epoch.BatchCount && Config.DumpEpochEvaluation))
                {
                    Config.EvaluationCallback!.Invoke(new DataSetEvaluation { Context = GetContext(), Result = evaluation });
                    Console.WriteLine($"Took {sw.Elapsed:m\\:ss\\.fff}");
                    sw.Restart();
                }
                batchCount++;
                Optimizer.OnBatchCompleted();
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

        //return new ModelTrainingResult
        //{
        //    EpochCount = Config.EpochCount,
        //    Before = before,
        //    After = EvaluateShort(),
        //};
        return null;
    }

    public ModelEvaluationResult EvaluateShort() => new()
    {
        TrainingSetResult = Evaluate(Config.GetRandomTrainingBatch()),
        TestSetResult = Evaluate(Config.GetRandomTestBatch()),
    };
    public DataSetEvaluationResult Evaluate(Batch<TInput, TOutput> batch)
    {
        int correctCounter = 0;
        double totalCost = 0;
        int totalCounter = 0;
        foreach(var entry in batch)
        {
            totalCounter++;
            var outputWeights = Network.Forward(Network.Embedder.Embed(entry.Input));
            var output = Network.Embedder.UnEmbed(outputWeights);

            if(output.Equals(entry.Expected))
            {
                correctCounter++;
            }

            totalCost += Config.Optimizer.CostFunction.TotalCost(outputWeights, Config.OutputResolver.Expected(entry.Expected));
        }

        return new()
        {
            TotalCount = totalCounter,
            CorrectCount = correctCounter,
            TotalCost = totalCost,
        };
    }
}

public static class ModelTrainer
{
    public static NetworkTrainer<TInput, TOutput> Create<TInput, TOutput>(SimpleNetwork<TInput, TOutput, RecordingLayer> model, TrainingConfig<TInput, TOutput> config) where TInput : notnull where TOutput : notnull
            => new(config, model);
}