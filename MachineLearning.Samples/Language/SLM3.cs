//using MachineLearning.Model.Layer;

//namespace MachineLearning.Samples.Language;

//public sealed class SLM3
//{
//    public const string TOKENS = " %'(),-.0123456789:=abcdefghijklmnopqrstuvwxyz\0";
//    public const int CONTEXT_SIZE = 128;

//    public static IOutputResolver<char> OutputResolver { get; } = new CharOutputResolver(TOKENS);
//    public static ModelSerializer Serializer { get; } = new(AssetManager.GetModelFile("sentences_3.gmw"));
//    public static FeedForwardModel<string, char> CreateModel(Random? random = null)
//    {
//        var initializer = new HeInitializer(random);
//        return AdvancedModelBuilder
//            .Create<StringEmbeddingLayer, string>(new StringEmbeddingLayer(TOKENS, CONTEXT_SIZE, 12), new StringEmbeddingLayer.Initializer(random))
//                .SetDefaultActivationFunction(LeakyReLUActivation.Instance)
//                .AddLayer(1024 * 2, initializer)
//                .AddLayer(1024 * 2, initializer)
//                .AddLayer(512 + 256, initializer)
//                .AddLayer(512, initializer)
//                .AddLayer(TOKENS.Length, new XavierInitializer(random), SoftMaxActivation.Instance)
//            .AddOutputLayer(new TokenOutputLayer(TOKENS, true, random));
//    }

//    public static IEmbeddedModel<string, char> TrainDefault(IEmbeddedModel<string, char>? model = null, TrainingConfig<string, char>? config = null, Random? random = null)
//    {
//        model ??= CreateModel(random);

//        var trainer = ModelTrainer.Generic(model, config ?? DefaultTrainingConfig());
//        trainer.TrainConsole();
//        Serializer.Save(model).Consume(
//            () => Console.WriteLine("Model saved!"),
//            flag => Console.WriteLine($"Error saving model: {flag}")
//        );
//        LMHelper.StartChat(model, CONTEXT_SIZE);
//        return model;
//    }

//    public static TrainingConfig<string, char> DefaultTrainingConfig(Random? random = null)
//    {
//        random ??= Random.Shared;

//        var dataSet = GetTrainingSet().ToArray();
//        random.Shuffle(dataSet);

//        var trainingSetSize = (int)(dataSet.Length * 0.9);

//        return new TrainingConfig<string, char>()
//        {
//            TrainingSet = dataSet,
//            TestSet = dataSet.Skip(trainingSetSize).ToArray(),

//            EpochCount = 32,
//            BatchCount = 256,

//            Optimizer = new AdamOptimizer
//            {
//                LearningRate = 0.01,
//                CostFunction = CrossEntropyLoss.Instance,
//            },

//            OutputResolver = OutputResolver,

//            EvaluationCallback = result => Console.WriteLine(result.Dump()),
//            DumpEvaluationAfterBatches = 1,

//            RandomSource = random,
//        };
//    }

//    public static IEnumerable<DataEntry<string, char>> GetTrainingSet(Random? random = null)
//    {
//        Console.WriteLine("Analyzing Trainings Data...");
//        var lines = LanguageDataSource.GetLines(AssetManager.Sentences).ToArray();
//        Console.WriteLine($"Longest sentence {lines.Max(s => s.Length)} chars");
//        var tokensUsedBySource = new string([.. lines.SelectMany(s => s).Distinct().Order()]);
//        Console.WriteLine($"Source uses '{tokensUsedBySource}'");
//        tokensUsedBySource.Consume(t => OutputResolver.Expected(t));

//        Console.WriteLine(lines.SelectDuplicates().Dump('\n'));
//        return lines.InContextSize(CONTEXT_SIZE).ExpandPerChar();
//    }
//}
