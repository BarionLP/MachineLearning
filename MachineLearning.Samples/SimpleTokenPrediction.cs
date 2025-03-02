using MachineLearning.Data;
using MachineLearning.Mamba;
using MachineLearning.Samples.Language;

namespace MachineLearning.Samples;

public class SimpleTokenPrediction
{
    public const int CONTEXT_SIZE = 10;
    public static CharTokenizer Tokenizer { get; } = new("0123456789");
    public static Mamba2VectorModel CreateModel(Random? random = null)
    {
        var model = new Mamba2VectorModel(layerCount: 5, Tokenizer.TokenCount, CONTEXT_SIZE, stateDimensions: 12, embeddingDimensions: 16);

        new EmbeddingLayer.Initializer(random).Initialize(model.InputLayer);
        new UnEmbeddingLayer.Initializer(random).Initialize(model.OutputLayer);
        var initer = new Mamba2VectorLayer.Initializer(random);
        model.HiddenLayers.Consume(initer.Initialize);

        return model;
    }

    public static TrainingConfig DefaultTrainingConfig(Random? random = null) => new()
    {
        EpochCount = 100,
        Threading = ThreadingMode.Single,
        EvaluationCallback = r => Console.WriteLine(r.Dump()),
        DumpEvaluationAfterBatches = 1,
        RandomSource = random ?? Random.Shared,
        Optimizer = new AdamOptimizer
        {
            LearningRate = 0.015f,
            CostFunction = CrossEntropyFromSoftmaxLoss.Instance,
        },
    };

    public static ITrainingSet GetTrainingSet(Random? random = null)
    {
        var tokens = Tokenizer.Tokenize("0123456789");

        var trainingsData = ((int[])[.. tokens, .. tokens]).SlidingWindow(null, CONTEXT_SIZE).ToTrainingDataMatrix(Tokenizer.TokenCount, CONTEXT_SIZE, null);
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
