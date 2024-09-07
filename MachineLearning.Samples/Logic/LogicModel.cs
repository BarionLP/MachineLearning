using MachineLearning.Samples.Language;

namespace MachineLearning.Samples.Logic;

public static class LogicModel
{
    public const int CONTEXT_SIZE = 24;
    public const string TOKENS = "0123456789+-*/^;";

    public static StringEmbedder GetEmbedder() => new(CONTEXT_SIZE, TOKENS, false);
    public static EmbeddedModel<string, char> CreateModel(Random? random = null)
    {
        var initializer = new HeInitializer(random);
        return new ModelBuilder(CONTEXT_SIZE * 8)
            .SetDefaultActivationMethod(SigmoidActivation.Instance)
            .AddLayer(256, initializer)
            .AddLayer(TOKENS.Length, initializer, SoftmaxActivation.Instance)
            .Build(GetEmbedder());
    }

    public static TrainingConfig<string, char> GetTrainingConfig(Random? random = null)
    {
        return new TrainingConfig<string, char>()
        {
            TrainingSet = LogicalStatementSource.GenerateAdditionStatements(1024, random).ToArray(),
            TestSet = LogicalStatementSource.GenerateAdditionStatements(128, random).ToArray(),

            EpochCount = 16,
            BatchCount = 512*2,

            Optimizer = new AdamOptimizer
            {
                LearningRate = 0.1,
                CostFunction = CrossEntropyLoss.Instance,
            },

            OutputResolver = new CharOutputResolver(TOKENS),

            DumpEvaluationAfterBatches = 32,
            EvaluationCallback = result => Console.WriteLine(result.Dump()),

            ShuffleTrainingSetPerEpoch = true,
            RandomSource = random ?? Random.Shared,
        };
    }

    public static void Train(EmbeddedModel<string, char> model, TrainingConfig<string, char> config) {
        var trainer = ModelTrainer.Create(model, config);
        trainer.TrainConsoleCancelable();
    }

    public static void OpenChat(EmbeddedModel<string, char> model) {
        string input;
        do
        {
            input = Console.ReadLine() ?? string.Empty;
            if (string.IsNullOrEmpty(input))
            {
                return;
            }
            Console.CursorTop--;
            Console.CursorLeft = 0;
            Console.WriteLine(Complete(input, model));
        } while (true);
    }

    public static string Complete(string input, EmbeddedModel<string, char> model)
    {
        while (true)
        {
            var prediction = model.Process(input);
            input += prediction;
            if (prediction == ';')
            {
                break;
            }
        }

        return input;
    }
}
