using Ametrin.Guards;
using MachineLearning.Data;
using MachineLearning.Data.Entry;
using MachineLearning.Model;
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
    public ModelCachePool CachePool { get; }

    public Mamba2ModelTrainer(Mamba2Model model, TrainingConfig config, ITrainingSet trainingSet)
    {
        Config = config;
        TrainingSet = trainingSet;
        Model = model;
        Optimizer = config.Optimizer;
        LayerOptimizers = [.. Model.Layers.Select(Optimizer.CreateLayerOptimizer)];
        CachePool = new([.. Model.Layers]);
    }

    public DataSetEvaluationResult TrainAndEvaluate(IEnumerable<TrainingData> trainingBatch)
    {
        var timeStamp = Stopwatch.GetTimestamp();

        var context = ThreadedTrainer.Train(
            trainingBatch,
            CachePool,
            Config.Threading,
            (entry, context) =>
            {
                var data = Guard.Is<TrainingData<Vector, Vector>>(entry);
                var weights = Update(data, context.Gradients);

                context.TotalCount++;
                context.TotalCost += Config.Optimizer.CostFunction.TotalCost(weights, data.ExpectedWeights);
            }
        );

        Apply(context.Gradients);

        var evalutaion = new DataSetEvaluationResult()
        {
            TotalCount = context.TotalCount,
            CorrectCount = context.CorrectCount,
            TotalCost = context.TotalCost,
            TotalElapsedTime = Stopwatch.GetElapsedTime(timeStamp),
        };

        CachePool.Return(context.Gradients);
        return evalutaion;
    }

    private Vector Update(TrainingData<Vector, Vector> data, ImmutableArray<IGradients> gradients)
    {
        Debug.Assert(gradients.Length == Model.Layers.Length);

        using var marker = CachePool.RentSnapshots(out var rented);
        var snapshots = rented.Cast<Mamba2ScalarLayer.Snapshot>().ToImmutableArray();
        var result = Model.Process(data.InputValue, snapshots);

        var outGradient = Optimizer.CostFunction.Derivative(result, data.ExpectedWeights);
        NumericsDebug.AssertValidNumbers(outGradient);

        for (int layerIndex = LayerOptimizers.Length - 1; layerIndex >= 0; layerIndex--)
        {
            LayerOptimizers[layerIndex].Update(outGradient, snapshots[layerIndex], gradients[layerIndex]);
            outGradient = snapshots[layerIndex].GradientInput;
            NumericsDebug.AssertValidNumbers(outGradient);
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
    public ModelCachePool CachePool { get; }


    public Mamba2VectorModelTrainer(Mamba2VectorModel model, TrainingConfig config, ITrainingSet trainingSet)
    {
        Config = config;
        TrainingSet = trainingSet;
        Model = model;
        Optimizer = config.Optimizer;
        InputLayerOptimizer = Optimizer.CreateLayerOptimizer(Model.InputLayer);
        LayerOptimizers = [.. Model.HiddenLayers.Select(Optimizer.CreateLayerOptimizer)];
        OutputLayerOptimizer = Optimizer.CreateLayerOptimizer(Model.OutputLayer);
        CachePool = new([Model.InputLayer, .. Model.HiddenLayers, Model.OutputLayer]);
    }

    public DataSetEvaluationResult TrainAndEvaluate(IEnumerable<TrainingData> trainingBatch)
    {
        var timeStamp = Stopwatch.GetTimestamp();

        var context = ThreadedTrainer.Train(
            trainingBatch,
            CachePool,
            Config.Threading,
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


        var evalutaion = new DataSetEvaluationResult()
        {
            TotalCount = context.TotalCount,
            CorrectCount = context.CorrectCount,
            TotalCost = context.TotalCost,
            TotalElapsedTime = Stopwatch.GetElapsedTime(timeStamp),
        };

        CachePool.Return(context.Gradients);
        return evalutaion;
    }

    private (Matrix, int) Update(TrainingData<int[], int> data, ImmutableArray<IGradients> gradients)
    {
        Debug.Assert(gradients.Length == Model.HiddenLayers.Length + 2);
        using var marker = CachePool.RentSnapshots(out var snapshots);
        var inputSnapshot = snapshots[0];
        ReadOnlySpan<Mamba2VectorLayer.Snapshot> hiddenSnapshots = [.. snapshots.Skip(1).Take(Model.HiddenLayers.Length).Cast<Mamba2VectorLayer.Snapshot>()];
        var outputSnapshot = (UnEmbeddingLayer.Snapshot)snapshots[^1];

        var (weights, result) = Model.Process(data.InputValue, snapshots);

        var gradient = Optimizer.CostFunction.Derivative(weights.Storage, data.ExpectedWeights);
        NumericsDebug.AssertValidNumbers(gradient);

        OutputLayerOptimizer.Update(gradient, outputSnapshot, gradients[^1]);
        gradient = outputSnapshot.InputGradient.Storage;

        for (int layerIndex = LayerOptimizers.Length - 1; layerIndex >= 0; layerIndex--)
        {
            LayerOptimizers[layerIndex].Update(gradient, hiddenSnapshots[layerIndex], gradients[layerIndex + 1]);
            gradient = hiddenSnapshots[layerIndex].GradientInput.Storage;
            NumericsDebug.AssertValidNumbers(gradient);
        }

        InputLayerOptimizer.Update(gradient, inputSnapshot, gradients[0]);

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
