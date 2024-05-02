using Simple.Network;
using Simple.Training.Evaluation;

namespace Simple.Training;

public sealed class NetworkTrainer<TInput, TOutput>(TrainingConfig<TInput, TOutput> config, RecordingNetwork<TInput, TOutput> network) where TInput : notnull where TOutput : notnull{
    public TrainingConfig<TInput, TOutput> Config { get; } = config;
    public RecordingNetwork<TInput, TOutput> Network { get; } = network;
    internal NetworkTrainingContext<TInput, TOutput> Context { get; } = new(network, config.Optimizer, config.OutputResolver);

    public NetworkTrainingResult Train() {
        // for each epoch 
        // train on all batches
        // decay learnrate

        var before = EvaluateShort();
        Config.Optimizer.Init();

        foreach(var epochIndex in ..Config.EpochCount) {
            var epoch = Config.GetEpoch();
            var batchCount = 0;
            
            if(Config.DumpEpochEvaluation) CallEvaluate();

            foreach(var batch in epoch) {
                if(Config.DumpBatchEvaluation && batchCount % Config.DumpEvaluationAfterBatches == 0) {
                    CallEvaluate();
                }

                Context.Learn(batch);
                batchCount++;
            }

            Config.Optimizer.OnEpochCompleted();
        
            void CallEvaluate(){
                Config.EvaluationCallback!.Invoke(EvaluateShort(new() {
                    CurrentBatch = batchCount,
                    MaxBatch = epoch.BatchCount,
                    CurrentEpoch = epochIndex + 1,
                    MaxEpoch = Config.EpochCount,
                    LearnRate = Config.Optimizer.CurrentLearningRate,
                }));
            }
        }

        return new() { 
            EpochCount = Config.EpochCount,
            Before = before, 
            After = EvaluateShort(), 
        };
    }

    public NetworkEvaluation EvaluateShort(NetworkEvaluationContext context) => new() {
        Context = context,
        Result = EvaluateShort(),
    };
    public NetworkEvaluationResult EvaluateShort() => new() {
        TrainingSetResult = Evaluate(Config.GetRandomTrainingBatch()),
        TestSetResult = Evaluate(Config.GetRandomTestBatch()),
    };
    public DataSetEvaluationResult Evaluate(Batch<TInput, TOutput> batch) {
        int correctCounter = 0;
        Number totalCost = 0;
        int totalCounter = 0;
        foreach(var entry in batch) {
            totalCounter++;
            var output = Network.Process(entry.Input);

            if(output.Equals(entry.Expected)) {
                correctCounter++;
            }
            
            totalCost += Config.CostFunction.TotalCost(Network.LastOutputWeights, Config.OutputResolver.Expected(entry.Expected));
        }

        return new() {
            TotalCount = totalCounter,
            CorrectCount = correctCounter,
            TotalCost = totalCost,
        };
    }
}