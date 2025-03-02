using System.Collections.Immutable;
using System.Diagnostics;
using Ametrin.Guards;
using MachineLearning.Data;
using MachineLearning.Data.Entry;
using MachineLearning.Model.Layer.Snapshot;
using MachineLearning.Training;
using MachineLearning.Training.Evaluation;
using MachineLearning.Training.Optimization;

namespace MachineLearning.Mamba;

public sealed class Mamba2ModelTrainer : ITrainer<Mamba2Model>
{
    public TrainingConfig Config { get; }
    public ITrainingSet TrainingSet { get; }
    public Mamba2Model Model { get; }
    public Optimizer Optimizer { get; }
    public ImmutableArray<ILayerOptimizer> LayerOptimizers { get; }
    public ILayerOptimizer OutputLayerOptimizer => LayerOptimizers[^1];

    public Mamba2ModelTrainer(Mamba2Model model, TrainingConfig config, ITrainingSet trainingSet)
    {
        Config = config;
        TrainingSet = trainingSet;
        Model = model;
        Optimizer = config.Optimizer;
        LayerOptimizers = [.. Model.Layers.Select(Optimizer.CreateLayerOptimizer)];
    }

    public DataSetEvaluationResult TrainAndEvaluate(IEnumerable<TrainingData> trainingBatch)
    {
        var timeStamp = Stopwatch.GetTimestamp();

        var context = ThreadedTrainer.Train(
            trainingBatch,
            () => [.. Model.Layers.Select(l => l.CreateGradientAccumulator())],
            Config.MultiThread ? -1 : 1,
            (entry, context) =>
            {
                var data = Guard.Is<TrainingData<Vector, Vector>>(entry);
                var weights = Update(data, context.Gradients);

                context.TotalCount++;
                context.TotalCost += Config.Optimizer.CostFunction.TotalCost(weights, data.ExpectedWeights);
            }
        );

        Apply(context.Gradients);

        return new()
        {
            TotalCount = context.TotalCount,
            CorrectCount = context.CorrectCount,
            TotalCost = context.TotalCost,
            TotalElapsedTime = Stopwatch.GetElapsedTime(timeStamp),
        };
    }

    private Vector Update(TrainingData<Vector, Vector> data, ImmutableArray<IGradients> gradients)
    {
        Debug.Assert(gradients.Length == Model.Layers.Length);

        var snapshots = Model.Layers.Select(LayerSnapshots.Get).Cast<Mamba2ScalarLayer.Snapshot>().ToImmutableArray();
        var result = Model.Process(data.InputValue, snapshots);

        var outGradient = Optimizer.CostFunction.Derivative(result, data.ExpectedWeights);
        NumericsDebug.AssertValidNumbers(outGradient);

        for (int layerIndex = LayerOptimizers.Length - 1; layerIndex >= 0; layerIndex--)
        {
            LayerOptimizers[layerIndex].Update(outGradient, snapshots[layerIndex], gradients[layerIndex]);
            outGradient = snapshots[layerIndex].GradientInput;
            NumericsDebug.AssertValidNumbers(outGradient);
        }

        foreach (var (layer, snapshot) in Model.Layers.Zip(snapshots))
        {
            LayerSnapshots.Return(layer, snapshot);
        }

        return result;
    }

    private void Apply(ImmutableArray<IGradients> gradients) => LayerOptimizers.Zip(gradients).Consume(p => p.First.Apply(p.Second));
    public void FullReset() => LayerOptimizers.Consume(layer => layer.FullReset());
}

public sealed class Mamba2VectorModelTrainer : ITrainer<Mamba2VectorModel>
{
    public TrainingConfig Config { get; }
    public ITrainingSet TrainingSet { get; }
    public Mamba2VectorModel Model { get; }
    public Optimizer Optimizer { get; }
    public ILayerOptimizer InputLayerOptimizer { get; }
    public ImmutableArray<ILayerOptimizer> LayerOptimizers { get; }
    public ILayerOptimizer OutputLayerOptimizer { get; }

    public Mamba2VectorModelTrainer(Mamba2VectorModel model, TrainingConfig config, ITrainingSet trainingSet)
    {
        Config = config;
        TrainingSet = trainingSet;
        Model = model;
        Optimizer = config.Optimizer;
        InputLayerOptimizer = Optimizer.CreateLayerOptimizer(Model.InputLayer);
        LayerOptimizers = [.. Model.HiddenLayers.Select(Optimizer.CreateLayerOptimizer)];
        OutputLayerOptimizer = Optimizer.CreateLayerOptimizer(Model.OutputLayer);
    }

    public DataSetEvaluationResult TrainAndEvaluate(IEnumerable<TrainingData> trainingBatch)
    {
        var timeStamp = Stopwatch.GetTimestamp();

        var context = ThreadedTrainer.Train(
            trainingBatch,
            () => [Model.InputLayer.CreateGradientAccumulator(), .. Model.HiddenLayers.Select(l => l.CreateGradientAccumulator()), Model.OutputLayer.CreateGradientAccumulator()],
            Config.MultiThread ? -1 : 1,
            (entry, context) =>
            {
                var data = Guard.Is<TrainingData<int[], int>>(entry);
                var (weights, result) = Update(data, context.Gradients);

                if (result == data.ExpectedValue)
                {
                    context.CorrectCount++;
                }
                context.TotalCount++;
                context.TotalCost += Config.Optimizer.CostFunction.TotalCost(weights.Storage, data.ExpectedWeights);
            }
        );

        Apply(context.Gradients);

        return new()
        {
            TotalCount = context.TotalCount,
            CorrectCount = context.CorrectCount,
            TotalCost = context.TotalCost,
            TotalElapsedTime = Stopwatch.GetElapsedTime(timeStamp),
        };
    }

    private (Matrix, int) Update(TrainingData<int[], int> data, ImmutableArray<IGradients> gradients)
    {
        Debug.Assert(gradients.Length == Model.HiddenLayers.Length + 2);
        var inputSnapshot = LayerSnapshots.Get(Model.InputLayer);
        var snapshots = Model.HiddenLayers.Select(LayerSnapshots.Get).Cast<Mamba2VectorLayer.Snapshot>().ToImmutableArray();
        var outputSnapshot = (UnEmbeddingLayer.Snapshot)LayerSnapshots.Get(Model.OutputLayer);

        var (weights, result) = Model.Process(data.InputValue, [inputSnapshot, .. snapshots, outputSnapshot]);

        var gradient = Optimizer.CostFunction.Derivative(weights.Storage, data.ExpectedWeights);
        NumericsDebug.AssertValidNumbers(gradient);

        OutputLayerOptimizer.Update(gradient, outputSnapshot, gradients[^1]);
        gradient = outputSnapshot.GradientInput.Storage;

        for (int layerIndex = LayerOptimizers.Length - 1; layerIndex >= 0; layerIndex--)
        {
            LayerOptimizers[layerIndex].Update(gradient, snapshots[layerIndex], gradients[layerIndex + 1]);
            gradient = snapshots[layerIndex].GradientInput.Storage;
            NumericsDebug.AssertValidNumbers(gradient);
        }

        InputLayerOptimizer.Update(gradient, inputSnapshot, gradients[0]);

        LayerSnapshots.Return(Model.InputLayer, inputSnapshot);
        LayerSnapshots.Return(Model.OutputLayer, outputSnapshot);
        foreach (var (layer, snapshot) in Model.HiddenLayers.Zip(snapshots))
        {
            LayerSnapshots.Return(layer, snapshot);
        }

        return (weights, result);
    }

    private void Apply(ImmutableArray<IGradients> gradients)
    {
        InputLayerOptimizer.Apply(gradients[0]);
        LayerOptimizers.Zip(gradients.Skip(1).Take(LayerOptimizers.Length)).Consume(p => p.First.Apply(p.Second));
        OutputLayerOptimizer.Apply(gradients[^1]);
    }

    public void FullReset()
    {
        InputLayerOptimizer.FullReset();
        LayerOptimizers.Consume(layer => layer.FullReset());
        OutputLayerOptimizer.FullReset();
    }
}
