using MachineLearning.Data;
using MachineLearning.Mamba;
using MachineLearning.Samples.Language;

namespace MachineLearning.Samples;

public class SimpleTokenPrediction
{
    public static Mamba2VectorModel CreateModel(Random? random = null)
    {
        var model = new Mamba2VectorModel(layerCount: 4, tokenCount: 10, contextSize: 10, stateDimensions: 8, embeddingDimensions: 4);

        new EmbeddingLayer.Initializer(random).Initialize(model.InputLayer);
        new UnEmbeddingLayer.Initializer(random).Initialize(model.OutputLayer);
        var initer = new Mamba2VectorLayer.Initializer(random);
        model.HiddenLayers.Consume(initer.Initialize);

        return model;
    }

    public static TrainingConfig DefaultTrainingConfig(Random? random = null) => new()
    {
        EpochCount = 100,
        MultiThread = false,
        EvaluationCallback = r => Console.WriteLine(r.Dump()),
        DumpEvaluationAfterBatches = 1,
        RandomSource = new Random(42),
        Optimizer = new AdamOptimizer
        {
            LearningRate = 0.025f,
            CostFunction = CrossEntropyFromSoftmaxLoss.Instance,
        }
    };

    public static ITrainingSet GetTrainingSet(Random? random = null)
    {
        var tokens = "0123456789".Select(c => c - '0');

        var trainingsData = ((int[])[.. tokens, .. tokens]).SlidingWindow(null, 10).Where(d => d.Input.Length == 10).ToTrainingDataMatrix(tokenCount: 10, contextSize: 10);
        return new PredefinedTrainingSet(trainingsData)
        {
            BatchCount = 1
        };
    }

    public static Mamba2VectorModel TrainDefault(Mamba2VectorModel? model = null, TrainingConfig? config = null, ITrainingSet? trainingSet = null, Random? random = null)
    {
        model ??= CreateModel(random);
        config ??= DefaultTrainingConfig(random);
        trainingSet ??= GetTrainingSet(random);

        var trainer = new Mamba2VectorModelTrainer(model, config, trainingSet);

        trainer.TrainConsole();

        return model;
    }
}
