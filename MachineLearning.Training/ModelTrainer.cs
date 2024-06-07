﻿using MachineLearning.Model;
using MachineLearning.Model.Layer;
using MachineLearning.Training.Evaluation;
using MachineLearning.Training.Optimization;
using System.Collections.Concurrent;
using System.Collections.Immutable;
using System.Diagnostics;

namespace MachineLearning.Training;

public sealed class ModelTrainer<TInput, TOutput> where TInput : notnull where TOutput : notnull
{
    public TrainingConfig<TInput, TOutput> Config { get; }
    public EmbeddedModel<TInput, TOutput> Model { get; }
    public IOptimizer Optimizer { get; }
    internal ModelTrainingContext<TInput, TOutput> Context { get; }

    public ModelTrainer(EmbeddedModel<TInput, TOutput> model, TrainingConfig<TInput, TOutput> config)
    {
        Config = config;
        Model = model;
        Optimizer = config.Optimizer.CreateOptimizer();
        Context = new(model, config, Optimizer);
    }

    public void Train(CancellationToken? token = null)
    {
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
                var evaluation = Context.TrainAndEvaluate(batch, true);
                if((Config.DumpBatchEvaluation && batchCount % Config.DumpEvaluationAfterBatches == 0) || (batchCount + 1 == epoch.BatchCount && Config.DumpEpochEvaluation))
                {
                    Config.EvaluationCallback!.Invoke(new DataSetEvaluation { Context = GetContext(), Result = evaluation });
                    Console.WriteLine($"Took {sw.Elapsed:m\\:ss\\.fff}");
                    sw.Restart();
                }
                batchCount++;
                Optimizer.OnBatchCompleted();

                if(token?.IsCancellationRequested is true){
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

        //return new ModelTrainingResult
        //{
        //    EpochCount = Config.EpochCount,
        //    Before = before,
        //    After = EvaluateShort(),
        //};
    }

    public ModelEvaluationResult EvaluateShort() => new()
    {
        TrainingSetResult = Evaluator.Evaluate(Model, Config.Optimizer.CostFunction, Config.OutputResolver, Config.GetRandomTrainingBatch()),
        TestSetResult = Evaluator.Evaluate(Model, Config.Optimizer.CostFunction, Config.OutputResolver, Config.GetRandomTestBatch()),
    };
}

public static class ModelTrainer
{
    public static ModelTrainer<TInput, TOutput> Create<TInput, TOutput>(EmbeddedModel<TInput, TOutput> model, TrainingConfig<TInput, TOutput> config) where TInput : notnull where TOutput : notnull
            => new(model, config);
}

public sealed class LayerSnapshot(int inputNodes, int outputNodes)
{
    private static readonly ConcurrentDictionary<SimpleLayer, ConcurrentQueue<LayerSnapshot>> _registry = [];
    public Vector LastRawInput = Vector.Empty;
    public readonly Vector LastWeightedInput = Vector.Create(outputNodes);
    public readonly Vector LastActivatedWeights = Vector.Create(outputNodes);
    public readonly Matrix WeightGradients = Matrix.Create(outputNodes, inputNodes);
    //private bool _isUsed = true;

    //private static readonly object _lock = new();
    public static LayerSnapshot Get(SimpleLayer layer)
    {
        var queue = _registry.GetOrAdd(layer, static (layer) => []);
        
        if (queue.TryDequeue(out var snapshot))
        {
            //Debug.Assert(!snapshot._isUsed);
            //snapshot._isUsed = true;
            return snapshot;
        }

        return new LayerSnapshot(layer.InputNodeCount, layer.OutputNodeCount);
    }

    public static void Return(SimpleLayer layer, LayerSnapshot snapshot)
    {
        //Debug.Assert(snapshot._isUsed);
        //snapshot.LastWeightedInput.ResetZero();
        //snapshot.LastActivatedWeights.ResetZero();
        //snapshot.WeightGradients.ResetZero();
        //snapshot._isUsed = false;
        _registry[layer].Enqueue(snapshot);
    }
}

public static class LayerExtensions
{
    public static Vector Forward(this SimpleLayer layer, Vector input, LayerSnapshot snapshot)
    {
        snapshot.LastRawInput = input;
        layer.Weights.Multiply(input, snapshot.LastWeightedInput);
        snapshot.LastWeightedInput.AddInPlace(layer.Biases);

        layer.ActivationFunction.Activate(snapshot.LastWeightedInput, snapshot.LastActivatedWeights);

        return snapshot.LastActivatedWeights;
    }

    public static Vector Forward(this SimpleModel model, Vector weights, ImmutableArray<LayerSnapshot> snapshots)
    {
        foreach(var (layer, snapshot) in model.Layers.Zip(snapshots))
        {
            weights = layer.Forward(weights, snapshot);
        }
        return weights;
    }
}