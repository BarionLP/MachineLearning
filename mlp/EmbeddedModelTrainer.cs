using Ametrin.Guards;
using MachineLearning.Data;
using MachineLearning.Data.Entry;
using MachineLearning.Model;
using MachineLearning.Model.Layer.Snapshot;
using MachineLearning.Training;
using MachineLearning.Training.Evaluation;
using MachineLearning.Training.Optimization;

namespace ML.MultiLayerPerceptron;

public sealed class EmbeddedModelTrainer<TIn, TOut> : ITrainer<EmbeddedModel<TIn, TOut>>
{
    public TrainingConfig Config { get; }
    public ITrainingSet TrainingSet { get; }
    public EmbeddedModel<TIn, TOut> Model { get; }
    public Optimizer Optimizer { get; }
    public ImmutableArray<ILayerOptimizer> LayerOptimizers { get; }
    public ILayerOptimizer OutputLayerOptimizer => LayerOptimizers[^1];
    public ModelCachePool CachePool { get; }

    public EmbeddedModelTrainer(EmbeddedModel<TIn, TOut> model, TrainingConfig config, ITrainingSet trainingSet)
    {
        Config = config;
        TrainingSet = trainingSet;
        Model = model;
        Optimizer = config.Optimizer;
        LayerOptimizers = [.. Model.InnerModel.Layers.Select(Optimizer.CreateLayerOptimizer)];
        CachePool = new([Model.InputLayer, .. Model.InnerModel.Layers, Model.OutputLayer]);
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
                var data = Guard.Is<TrainingData<TIn, TOut>>(entry);
                var weights = Update(data, context.Gradients);

                if (Model.OutputLayer.Process(weights).output!.Equals(data.ExpectedValue))
                {
                    context.CorrectCount++;
                }

                context.TotalCount++;
                context.TotalCost += Config.Optimizer.CostFunction.TotalCost(weights, data.ExpectedWeights);
            }
        );

        Apply(context.Gradients);

        var evaluation = new DataSetEvaluationResult()
        {
            TotalCount = context.TotalCount,
            CorrectCount = context.CorrectCount,
            TotalCost = context.TotalCost,
            TotalElapsedTime = Stopwatch.GetElapsedTime(timeStamp),
        };

        CachePool.Return(context.Gradients);

        return evaluation;
    }

    private Vector Update(TrainingData<TIn, TOut> data, ImmutableArray<IGradients> gradients)
    {
        Debug.Assert(gradients.Length == Model.InnerModel.Layers.Length + 2);

        using var marker = CachePool.RentSnapshots(out var rented);
        var snapshots = rented.Skip(1).Take(Model.InnerModel.Layers.Length).Cast<PerceptronLayer.Snapshot>().ToImmutableArray();
        var inputWeights = Model.InputLayer.Process(data.InputValue);
        var result = Model.InnerModel.Process(inputWeights, snapshots);

        var nodeValues = Config.Optimizer.CostFunction.Derivative(result, data.ExpectedWeights);
        NumericsDebug.AssertValidNumbers(nodeValues);
        OutputLayerOptimizer.Update(nodeValues, snapshots[^1], gradients[^2]);
        nodeValues = snapshots[^1].InputGradient;
        NumericsDebug.AssertValidNumbers(nodeValues);

        for (int hiddenLayerIndex = LayerOptimizers.Length - 2; hiddenLayerIndex >= 0; hiddenLayerIndex--)
        {
            var hiddenLayer = LayerOptimizers[hiddenLayerIndex];
            hiddenLayer.Update(nodeValues, snapshots[hiddenLayerIndex], gradients[hiddenLayerIndex + 1]);
            nodeValues = snapshots[hiddenLayerIndex].InputGradient;
            NumericsDebug.AssertValidNumbers(nodeValues);
        }

        return result;
    }

    private void Apply(ImmutableArray<IGradients> gradients) => LayerOptimizers.Zip(gradients.Skip(1).Take(Model.InnerModel.Layers.Length)).Consume(p => p.First.Apply(p.Second));
    public void FullReset() => LayerOptimizers.Consume(layer => layer.FullReset());
}
