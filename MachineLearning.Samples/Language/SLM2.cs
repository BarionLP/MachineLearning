using MachineLearning.Serialization;
using static System.Runtime.InteropServices.JavaScript.JSType;
using System.Runtime.ConstrainedExecution;
using System.Runtime.Intrinsics.X86;
using System;

namespace MachineLearning.Samples.Language;

public sealed class SLM2 : ISample<string, char>
{
    public const int CONTEXT_SIZE = 128;
    public const string TOKENS = " %'(),-.0123456789:=abcdefghijklmnopqrstuvwxyz";
    public static IEmbedder<string, char> Embedder { get; } = new StringEmbedder(CONTEXT_SIZE, TOKENS, true);
    public static IOutputResolver<char> OutputResolver { get; } = new CharOutputResolver(TOKENS);
    public static ModelSerializer Serializer { get; } = new(AssetManager.GetModelFile("sentence_2.nnw"));

    public static EmbeddedModel<string, char> CreateModel(Random? random = null)
    {
        var initializer = new HeInitializer(random);
        return new ModelBuilder(CONTEXT_SIZE * 8)
            .SetDefaultActivationMethod(LeakyReLUActivation.Instance)
            .AddLayer(1024 * 2, initializer)
            .AddLayer(1024 * 2, initializer)
            .AddLayer(512 + 256, initializer)
            .AddLayer(512, initializer)
            .AddLayer(TOKENS.Length, new XavierInitializer(random), SoftmaxActivation.Instance)
            .Build(Embedder);
    }

    public static TrainingConfig<string, char> DefaultTrainingConfig(Random? random = null)
    {
        random ??= Random.Shared;

        var dataSet = GetTrainingSet().ToArray();
        random.Shuffle(dataSet);

        var trainingSetSize = (int) (dataSet.Length * 0.9);

        return new TrainingConfig<string, char>()
        {
            TrainingSet = dataSet,
            TestSet = dataSet.Skip(trainingSetSize).ToArray(),

            EpochCount = 32,
            BatchCount = 256 + 128,

            Optimizer = new AdamOptimizer
            {
                LearningRate = 0.01,
                CostFunction = CrossEntropyLoss.Instance,
            },

            OutputResolver = OutputResolver,

            EvaluationCallback = result => Console.WriteLine(result.Dump()),
            DumpEvaluationAfterBatches = 32,

            RandomSource = random,
        };
    }

    public static IEnumerable<DataEntry<string, char>> GetTrainingSet(Random? random = null)
    {
        Console.WriteLine("Analyzing Training Data...");
        var lines = LanguageDataSource.GetLines(AssetManager.Sentences).ToArray();
        lines.ForEach(l => Embedder.Embed(l));
        Console.WriteLine($"Longest sentence {lines.Max(s => s.Length)}");
        var tokensUsedBySource = new string(lines.SelectMany(s => s).Distinct().Order().ToArray());
        Console.WriteLine($"Source uses '{tokensUsedBySource}'");
        tokensUsedBySource.ForEach(t => OutputResolver.Expected(t));

        Console.WriteLine(lines.SelectDuplicates().Dump('\n'));
        return lines.InContextSize(CONTEXT_SIZE).ExpandPerChar();
    }

    public static EmbeddedModel<string, char> TrainDefault(EmbeddedModel<string, char>? model = null, TrainingConfig<string, char>? trainingConfig = null, Random? random = null)
    {
        model ??= Serializer.Load(Embedder).ReduceOrThrow();

        SimpleLM.TrainDefault(model, trainingConfig ?? DefaultTrainingConfig(), random);
        Serializer.Save(model);
        Console.WriteLine("Model saved!");
        SimpleLM.StartChat(model);
        return model;
    }
}