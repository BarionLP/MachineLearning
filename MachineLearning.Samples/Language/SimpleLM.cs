using ModelDefinition = MachineLearning.Model.EmbeddedModel<string, char>;

namespace MachineLearning.Samples.Language;

public static class SimpleLM
{
    public const int ContextSize = 256 + 64;
    public static ModelDefinition GetModel(Random? random = null)
    {
        var initializer = new HeInitializer(random);
        return new ModelBuilder(ContextSize * 8)
            .SetDefaultActivationMethod(LeakyReLUActivation.Instance)
            .AddLayer(2048, initializer)
            .AddLayer(512, initializer)
            .AddLayer(128, initializer)
            .AddLayer(LanguageDataSource.TOKENS.Length, builder => builder.Initialize(new XavierInitializer(random)).SetActivationMethod(SoftmaxActivation.Instance))
            .Build(new StringEmbedder(ContextSize));
    }

    public static TrainingConfig<string, char> GetTrainingConfig(Random? random = null)
    {
        random ??= Random.Shared;
        var dataSet = LanguageDataSource.GetLines(AssetManager.Speech).InContextSize(ContextSize).ExpandPerChar().ToArray();
        random.Shuffle(dataSet);

        var trainingSetSize = (int) (dataSet.Length * 0.9);
        return new TrainingConfig<string, char>()
        {
            TrainingSet = dataSet.Take(trainingSetSize).ToArray(),
            TestSet = dataSet.Skip(trainingSetSize).ToArray(),

            EpochCount = 8,
            BatchCount = 256,

            Optimizer = new AdamOptimizerConfig
            {
                LearningRate = 0.01,
                CostFunction = CrossEntropyLoss.Instance,
            },

            OutputResolver = new CharOutputResolver(),

            EvaluationCallback = result => Console.WriteLine(result.Dump()),
            DumpEvaluationAfterBatches = 32,

            RandomSource = random,
            ShuffleTrainingSetPerEpoch = true,
        };
    }

    public static ModelDefinition TrainDefault(Random? random = null) => TrainDefault(GetModel(random ?? Random.Shared), random);
    public static ModelDefinition TrainDefault(ModelDefinition model, Random? random = null)
    {
        var config = GetTrainingConfig(random ?? Random.Shared);
        var trainer = ModelTrainer.Create(model, config);
        using var cts = new CancellationTokenSource();
        Task.Run(() =>
        {
            while(!cts.IsCancellationRequested)
            {
                if(Console.KeyAvailable && Console.ReadKey(intercept: true).Key == ConsoleKey.C)
                {
                    Console.WriteLine("Canceling...");
                    cts.Cancel();
                    break;
                }
                Thread.Sleep(500);
            }
        });

        //cts.CancelAfter(TimeSpan.FromSeconds(30));
        Console.WriteLine("Starting Training...");
        trainer.Train(cts.Token);
        cts.Cancel();
        Console.WriteLine("Training Done!");

        Generate("Männer, ", model);

        return model;
    }

    private const string EndSymbols = ".!?";
    public static void Generate(string input, ModelDefinition model)
    {
        input = input.ToLowerInvariant();
        Console.Write(input);
        char prediction;
        do
        {
            prediction = model.Process(input);
            input += prediction;
            Console.Write(prediction);
        } while(!EndSymbols.Contains(prediction) && input.Length < ContextSize);
        Console.WriteLine();
    }

    public static void StartChat(ModelDefinition model)
    {
        string input;
        do
        {
            input = Console.ReadLine() ?? string.Empty;
            if(string.IsNullOrEmpty(input))
            {
                return;
            }
            Generate(input, model);
        } while(true);
    }
}
