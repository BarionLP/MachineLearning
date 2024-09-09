using ModelDefinition = MachineLearning.Model.EmbeddedModel<string, char>;

namespace MachineLearning.Samples.Language;

public sealed class SimpleLM : ISample<string, char>
{
    public const int CONTEXT_SIZE = 256 + 64;
    public static ModelDefinition CreateModel(Random? random = null)
    {
        var initializer = new HeInitializer(random);
        return new ModelBuilder(CONTEXT_SIZE * 8)
            .SetDefaultActivationMethod(LeakyReLUActivation.Instance)
            .AddLayer(2048, initializer)
            .AddLayer(512, initializer)
            .AddLayer(128, initializer)
            .AddLayer(LanguageDataSource.TOKENS.Length, builder => builder.Initialize(new XavierInitializer(random)).SetActivationMethod(SoftmaxActivation.Instance))
            .Build(new BinaryStringEmbedder(CONTEXT_SIZE, LanguageDataSource.TOKENS, true));
    }

    public static TrainingConfig<string, char> DefaultTrainingConfig(Random? random = null)
    {
        random ??= Random.Shared;
        var dataSet = GetTrainingSet().ToArray();
        random.Shuffle(dataSet);

        var trainingSetSize = (int)(dataSet.Length * 0.9);
        return new TrainingConfig<string, char>()
        {
            TrainingSet = dataSet.Take(trainingSetSize).ToArray(),
            TestSet = dataSet.Skip(trainingSetSize).ToArray(),

            EpochCount = 8,
            BatchCount = 256,

            Optimizer = new AdamOptimizer
            {
                LearningRate = 0.01,
                CostFunction = CrossEntropyLoss.Instance,
            },

            OutputResolver = new CharOutputResolver(LanguageDataSource.TOKENS),

            EvaluationCallback = result => Console.WriteLine(result.Dump()),
            DumpEvaluationAfterBatches = 32,

            RandomSource = random,
            ShuffleTrainingSetPerEpoch = true,
        };
    }

    public static ModelDefinition TrainDefault(ModelDefinition? model = null, TrainingConfig<string, char>? config = null, Random? random = null)
    {
        model ??= CreateModel(random);
        config ??= DefaultTrainingConfig(random);

        var trainer = ModelTrainer.Create(model, config);
        trainer.TrainConsole();

        return model;
    }

    private const string EndSymbols = ".!?";
    public static void Generate(string input, ModelDefinition model)
    {
        input = input.ToLowerInvariant();
        Console.Write(input);
        char prediction;
        Weight confidence;
        do
        {
            (prediction, confidence) = model.Process(input);
            input += prediction;
            SetConsoleTextColor(confidence);
            Console.Write(prediction);
        } while (!EndSymbols.Contains(prediction) && input.Length < CONTEXT_SIZE);
        Console.Write("\u001b[0m"); //reset color
        Console.WriteLine();

        static void SetConsoleTextColor(double confidence)
        {
            Console.Write($"\u001b[38;2;{(1-confidence)*255:F0};{confidence*255:F0};60m");
        }
    }

    public static void StartChat(ModelDefinition model)
    {
        string input;
        do
        {
            input = Console.ReadLine() ?? string.Empty;
            if (string.IsNullOrEmpty(input))
            {
                return;
            }
            Generate(input, model);
        } while (true);
    }

    public static IEnumerable<DataEntry<string, char>> GetTrainingSet(Random? random = null) => LanguageDataSource.GetLines(AssetManager.Speech).InContextSize(CONTEXT_SIZE).ExpandPerChar();
}
