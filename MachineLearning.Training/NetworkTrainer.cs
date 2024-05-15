using MachineLearning.Model;
using MachineLearning.Training.Evaluation;
using MachineLearning.Training.Optimization;

namespace MachineLearning.Training;

public sealed class NetworkTrainer<TInput, TOutput> where TInput : notnull where TOutput : notnull
{
    public TrainingConfig<TInput, TOutput> Config { get; }
    public RecordingNetwork<TInput, TOutput> Network { get; }
    public IOptimizer<double> Optimizer { get; }
    internal NetworkTrainingContext<TInput, TOutput> Context { get; }

    public NetworkTrainer(TrainingConfig<TInput, TOutput> config, RecordingNetwork<TInput, TOutput> network)
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
        // decay learnrate

        var before = EvaluateShort();
        Optimizer.Init();
        Context.FullReset();
        foreach (var epochIndex in ..Config.EpochCount)
        {
            var epoch = Config.GetEpoch();
            var batchCount = 0;

            foreach (var batch in epoch)
            {

                var evaluation = Context.TrainAndEvaluate(batch);
                if ((Config.DumpBatchEvaluation && batchCount % Config.DumpEvaluationAfterBatches == 0) || (batchCount+1 == epoch.BatchCount && Config.DumpEpochEvaluation))
                {
                    Config.EvaluationCallback!.Invoke(new DataSetEvaluation { Context = GetContext(), Result = evaluation });
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

        return new ModelTrainingResult
        {
            EpochCount = Config.EpochCount,
            Before = before,
            After = EvaluateShort(),
        };
    }

    public ModelEvaluationResult EvaluateShort() => new()
    {
        TrainingSetResult = Evaluate(Config.GetRandomTrainingBatch()),
        TestSetResult = Evaluate(Config.GetRandomTestBatch()),
    };
    public DataSetEvaluationResult Evaluate(Batch<TInput, TOutput> batch)
    {
        int correctCounter = 0;
        Number totalCost = 0;
        int totalCounter = 0;
        foreach (var entry in batch)
        {
            totalCounter++;
            var output = Network.Process(entry.Input);

            if (output.Equals(entry.Expected))
            {
                correctCounter++;
            }

            totalCost += Config.Optimizer.CostFunction.TotalCost(Network.LastOutputWeights, Config.OutputResolver.Expected(entry.Expected));
        }

        return new()
        {
            TotalCount = totalCounter,
            CorrectCount = correctCounter,
            TotalCost = totalCost,
        };
    }
}