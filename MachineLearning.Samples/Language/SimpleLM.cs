﻿using ModelDefinition = MachineLearning.Model.EmbeddedModel<string, char>;

namespace MachineLearning.Samples.Language;

public static class SimpleLM
{
    public const int CONTEXT_SIZE = 256 + 64;
    public static ModelDefinition GetModel(Random? random = null)
    {
        var initializer = new HeInitializer(random);
        return new ModelBuilder(CONTEXT_SIZE * 8)
            .SetDefaultActivationMethod(LeakyReLUActivation.Instance)
            .AddLayer(2048, initializer)
            .AddLayer(512, initializer)
            .AddLayer(128, initializer)
            .AddLayer(LanguageDataSource.TOKENS.Length, builder => builder.Initialize(new XavierInitializer(random)).SetActivationMethod(SoftmaxActivation.Instance))
            .Build(new StringEmbedder(CONTEXT_SIZE, LanguageDataSource.TOKENS, true));
    }

    public static TrainingConfig<string, char> GetTrainingConfig(Random? random = null)
    {
        random ??= Random.Shared;
        var dataSet = LanguageDataSource.GetLines(AssetManager.Speech).InContextSize(CONTEXT_SIZE).ExpandPerChar().ToArray();
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

            OutputResolver = new CharOutputResolver(LanguageDataSource.TOKENS),

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
        trainer.TrainConsoleCancelable();

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
        } while(!EndSymbols.Contains(prediction) && input.Length < CONTEXT_SIZE);
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
