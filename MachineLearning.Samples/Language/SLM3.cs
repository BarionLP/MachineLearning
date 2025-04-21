using MachineLearning.Data;
using MachineLearning.Model.Layer;
using ML.MultiLayerPerceptron;
using ML.MultiLayerPerceptron.Initialization;

namespace MachineLearning.Samples.Language;

public static class SLM3
{
    // Console.WriteLine(string.Join(", ",
    //     dataSet.GetLines().Take(20000)
    //     .SelectMany(l => l.Split([' ', '.', ',', '!', '?', ';', ':'], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
    //     .Select(s => s.Replace('’', '\''))
    //     .CountBy(w => w, StringComparer.InvariantCultureIgnoreCase).OrderByDescending(g => g.Value).Select(g => g.Key).Where(s => s.Length > 1).Select(s => $"\"{s.ToLower()}\"").Take(255-SLM3.SYMBOLS.Length)
    // ));
    public static readonly HashSet<string> WORD_TOKENS = ["the", "and", "to", "of", "in", "is", "for", "that", "you", "with", "on", "it", "are", "as", "this", "be", "your", "or", "have", "at", "was", "from", "we", "by", "will", "not", "can", "an", "but", "all", "they", "if", "has", "our", "my", "more", "their", "one", "so", "he", "about", "which", "when", "what", "also", "out", "his", "up", "there", "time", "new", "do", "who", "like", "some", "other", "been", "just", "get", "how", "her", "would", "had", "them", "were", "any", "no", "these", "into", "me", "than", "people", "its", "make", "most", "only", "may", "she", "us", "first", "over", "use", "work", "very", "after", "well", "then", "now", "many", "need", "even", "through", "way", "two", "good", "best", "because", "see", "years", "know", "where", "day", "should", "much", "could", "such", "great", "here", "while", "take", "help", "home", "said", "back", "want", "it's", "being", "before", "year", "those", "find", "each", "made", "right", "used", "life", "go", "world", "free", "information", "business", "really", "every", "love", "think", "own", "both", "still", "around", "him", "last", "going", "off", "same", "different", "look", "place", "part", "between", "too", "did", "down", "service", "am", "during", "does", "since", "using", "high", "things", "company", "always", "another", "few", "set", "little", "available", "long", "services", "without", "online", "don't", "system", "family", "experience", "something", "come", "data", "next", "school", "why", "better", "sure", "under", "give", "however", "must", "including", "support", "can't"];
    public const string SYMBOLS = "\0 ?!\"#$%&'()*+,-./0123456789:;=?_abcdefghijklmnopqrstuvwxyz|ßäöü€";
    public const int CONTEXT_SIZE = 128 + 64;
    public static StringTokenizer Tokenizer { get; } = new(WORD_TOKENS, SYMBOLS, [("“", "\""), ("”", "\""), ("\n", " "), ("–", "-"), ("—", "-"), ("’", "'"), ("it’s", "it's"), ("don’t", "don't"), ("can’t", "can't")]);
    public static ModelSerializer Serializer { get; } = new(AssetManager.GetModelFile("slm3"));
    public static EmbeddedModel<int[], int> CreateModel(Random? random = null)
    {
        var initializer = new HeInitializer(random);
        return EmbeddedModelBuilder
            .Create(new EncodedEmbeddingLayer(Tokenizer.TokenCount, CONTEXT_SIZE))
                .DefaultActivation(LeakyReLUActivation.Instance)
                .AddLayer(1024 * 2, initializer)
                .AddLayer(1024 * 2 , initializer)
                .AddLayer(1024, initializer)
                .AddLayer(1024, initializer)
                .AddLayer(Tokenizer.TokenCount, new XavierInitializer(random), SoftMaxActivation.Instance)
            .AddOutputLayer(new TokenOutputLayer(Tokenizer.TokenCount, true, random));
    }

    public static EmbeddedModel<int[], int> TrainDefault(EmbeddedModel<int[], int>? model = null, TrainingConfig? config = null, ITrainingSet? trainingSet = null, Random? random = null)
    {
        model ??= Serializer.Load<EmbeddedModel<int[], int>>().Or(error =>
        {
            Console.WriteLine("No existing model found! Creating new!");
            return CreateModel(random);
        });

        var trainer = new EmbeddedModelTrainer<int[], int>(model, config ?? DefaultTrainingConfig(), trainingSet ?? GetTrainingSet());
        trainer.TrainConsole();
        Serializer.Save(model).Consume(
            () => Console.WriteLine("Model saved!"),
            error => Console.WriteLine($"Error saving model: {error.Message}")
        );
        LMHelper.StartChat(model, CONTEXT_SIZE, Tokenizer);
        return model;
    }

    public static TrainingConfig DefaultTrainingConfig(Random? random = null) => new()
    {
        EpochCount = 32,

        Optimizer = new AdamOptimizer
        {
            LearningRate = 0.0001f,
            CostFunction = CrossEntropyLoss.Instance,
        },

        EvaluationCallback = result => Console.WriteLine(result.Dump()),
        DumpEvaluationAfterBatches = 16,
        //MultiThread = false,
        RandomSource = random ?? Random.Shared,
    };

    public static ITrainingSet GetTrainingSet(Random? random = null)
    {
        Console.WriteLine("Analyzing Trainings Data...");
        var lines = LanguageDataHelper.GetLines(AssetManager.Sentences).ToArray();
        Console.WriteLine($"Longest sentence {lines.Max(s => s.Length)} chars");
        var tokensUsedBySource = new string([.. lines.SelectMany(s => s).Distinct().Order()]);
        Console.WriteLine($"Source uses '{tokensUsedBySource}'");

        Console.WriteLine(lines.SelectDuplicates().Dump('\n'));

        //var words = lines.SelectMany(l => l.Split([' ', '.', ','], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries));
        //var usages = words.CountBy(w => w).OrderByDescending(g => g.Value).Select(g => $"{g.Key}: {g.Value}");
        //Console.WriteLine(string.Join('\n', usages.Take(50)));
        var endToken = Tokenizer.TokenizeSingle("\0");

        return new PredefinedTrainingSet(lines.Tokenize(Tokenizer).ExpandPerToken(endToken, CONTEXT_SIZE).ToTrainingData(Tokenizer.TokenCount)) 
        {
            BatchCount = 256,
            Random = random ?? Random.Shared,
        };
    }
}
