// using System.Collections.Immutable;
// using System.Diagnostics;
// using System.Formats.Tar;
// using MachineLearning.Data;
// using MachineLearning.Data.Entry;
// using MachineLearning.Model;
// using MachineLearning.Model.Layer.Snapshot;
// using MachineLearning.Training;
// using MachineLearning.Training.Cost;
// using MachineLearning.Training.Evaluation;
// using MachineLearning.Training.Optimization;

// namespace MachineLearning.Mamba;

// public sealed class GenericModelTrainer<TModel, TArch, TSnapshot> : ITrainer<TModel> 
// where TModel : IModel<TArch, TSnapshot>
// where TSnapshot : ILayerSnapshot
// {
//     public TrainingConfig Config { get; }
//     public ITrainingSet TrainingSet { get; }
//     public ICostFunction<TArch> CostFunction { get; }
//     public TModel Model { get; }
//     public Optimizer Optimizer { get; }
//     public ImmutableArray<ILayerOptimizer> LayerOptimizers { get; }

//     public GenericModelTrainer(TModel model, TrainingConfig config, ITrainingSet trainingSet, ICostFunction<TArch> costFunction)
//     {
//         Config = config;
//         TrainingSet = trainingSet;
//         CostFunction = costFunction;
//         Model = model;
//         Optimizer = config.Optimizer;
//         LayerOptimizers = [.. Model.Layers.Select(Optimizer.CreateLayerOptimizer)];
//     }

//     public void Train(CancellationToken? token = null)
//     {
//         Optimizer.Init();
//         FullReset();
//         var cachedEvaluation = DataSetEvaluationResult.ZERO;
//         foreach (var (epochIndex, epoch) in TrainerHelper.GetEpochs(TrainingSet, Config.EpochCount).Index())
//         {
//             foreach (var (batchIndex, batch) in epoch.Index())
//             {
//                 cachedEvaluation += TrainAndEvaluate(batch.Cast<TrainingData<TArch>>());
//                 if (Config.DumpBatchEvaluation && batchIndex % Config.DumpEvaluationAfterBatches == 0 || batchIndex + 1 == epoch.BatchCount && Config.DumpEpochEvaluation)
//                 {
//                     Config.EvaluationCallback!.Invoke(new DataSetEvaluation { Context = GetContext(), Result = cachedEvaluation });
//                     cachedEvaluation = DataSetEvaluationResult.ZERO;
//                 }
//                 Optimizer.OnBatchCompleted();

//                 if (token?.IsCancellationRequested is true)
//                 {
//                     Optimizer.OnEpochCompleted();
//                     return;
//                 }

//                 TrainingEvaluationContext GetContext() => new()
//                 {
//                     CurrentBatch = batchIndex + 1,
//                     MaxBatch = epoch.BatchCount,
//                     CurrentEpoch = epochIndex + 1,
//                     MaxEpoch = Config.EpochCount,
//                     LearnRate = Config.Optimizer.LearningRate,
//                 };
//             }

//             Optimizer.OnEpochCompleted();
//         }
//     }

//     public DataSetEvaluationResult TrainAndEvaluate(IEnumerable<TrainingData<TArch>> trainingBatch)
//     {
//         var timeStamp = Stopwatch.GetTimestamp();
//         GradientCostReset();
//         int correctCounter = 0;
//         double totalCost = 0;
//         var dataCounter = 0;

//         if (Config.MultiThread)
//         {
//             var _lock = new Lock();
//             Parallel.ForEach(trainingBatch, (data) =>
//             {
//                 var weights = Update(data);

//                 lock (_lock)
//                 {
//                     UpdateCounters(data, weights);
//                 }
//             });
//         }
//         else
//         {
//             foreach (var data in trainingBatch)
//             {
//                 var weights = Update(data);
//                 UpdateCounters(data, weights);
//             }
//         }

//         void UpdateCounters(TrainingData<TArch> data, TArch weights)
//         {
//             dataCounter++;
//             totalCost += CostFunction.TotalCost(weights, data.Expected);
//         }

//         Apply(dataCounter);

//         return new()
//         {
//             TotalCount = dataCounter,
//             CorrectCount = correctCounter,
//             TotalCost = totalCost,
//             TotalElapsedTime = Stopwatch.GetElapsedTime(timeStamp),
//         };
//     }

//     private TArch Update(TrainingData<TArch> data)
//     {
//         var snapshots = Model.Layers.Select(LayerSnapshots.Get).Cast<TSnapshot>().ToImmutableArray();
//         var result = Model.Process(data.Input, snapshots);

//         var gradient = CostFunction.Derivative(result, data.Expected);
//         // NumericsDebug.AssertValidNumbers(gradient);

//         for (int layerIndex = LayerOptimizers.Length - 1; layerIndex >= 0; layerIndex--)
//         {
//             // LayerOptimizers[layerIndex].Update(gradient, snapshots[layerIndex]);
//             // gradient = snapshots[layerIndex].GradientInput;
//             // NumericsDebug.AssertValidNumbers(gradient);
//         }

//         foreach (var (layer, snapshot) in Model.Layers.Zip(snapshots))
//         {
//             LayerSnapshots.Return(layer, snapshot);
//         }

//         return result;
//     }

//     private void Apply(int dataCounter) => LayerOptimizers.Consume(layer => layer.Apply(dataCounter));
//     private void GradientCostReset() => LayerOptimizers.Consume(layer => layer.GradientCostReset());
//     private void FullReset() => LayerOptimizers.Consume(layer => layer.FullReset());
// }
