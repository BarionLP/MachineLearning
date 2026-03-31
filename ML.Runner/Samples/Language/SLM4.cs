using System.IO;
using ML.Core.Converters;
using ML.Core.Data.Training;
using ML.Core.Evaluation.Cost;
using ML.Core.Modules;
using ML.Core.Modules.Activations;
using ML.Core.Training;
using TrainingEntry = ML.Core.Data.Training.TrainingEntry<int[], Ametrin.Numerics.Matrix, int>;

namespace ML.Runner.Samples.Language;

public static class SLM4
{
    public static readonly HashSet<string> WORD_TOKENS = ["the", "and", "to", "of", "in", "is", "for", "that", "you", "with", "on", "it", "are", "as", "this", "be", "your", "or", "have", "at", "was", "from", "we", "by", "will", "not", "can", "an", "but", "all", "they", "if", "has", "our", "my", "more", "their", "one", "so", "he", "about", "which", "when", "what", "also", "out", "his", "up", "there", "time", "new", "do", "who", "like", "some", "other", "been", "just", "get", "how", "her", "would", "had", "them", "were", "any", "no", "these", "into", "me", "than", "people", "its", "make", "most", "only", "may", "she", "us", "first", "over", "use", "work", "very", "after", "well", "then", "now", "many", "need", "even", "through", "way", "two", "good", "best", "because", "see", "years", "know", "where", "day", "should", "much", "could", "such", "great", "here", "while", "take", "help", "home", "said", "back", "want", "it's", "being", "before", "year", "those", "find", "each", "made", "right", "used", "life", "go", "world", "free", "information", "business", "really", "every", "love", "think", "own", "both", "still", "around", "him", "last", "going", "off", "same", "different", "look", "place", "part", "between", "too", "did", "down", "service", "am", "during", "does", "since", "using", "high", "things", "company", "always", "another", "few", "set", "little", "available", "long", "services", "without", "online", "don't", "system", "family", "experience", "something", "come", "data", "next", "school", "why", "better", "sure", "under", "give", "however", "must", "including", "support", "can't"];
    public const string SYMBOLS = "\0 ?!\"#$%&'()*+,-./0123456789:;=?_abcdefghijklmnopqrstuvwxyz|ßäöü€";
    public const int CONTEXT_SIZE = int.MaxValue;
    public static StringTokenizer Tokenizer { get; } = new(WORD_TOKENS, SYMBOLS, [("“", "\""), ("”", "\""), ("\n", " "), ("–", "-"), ("—", "-"), ("’", "'"), ("it’s", "it's"), ("don’t", "don't"), ("can’t", "can't")], StringComparer.InvariantCultureIgnoreCase);
    public static FileInfo ModelFile { get; } = AssetManager.GetModelFile("slm4");

    public static EmbeddedModule<int[], Matrix, int> CreateAndInitModel(Random random)
    {
        const int EMBEDDING_SIZE = 92;
        const int HIDDEN_SIZE = 512;
        var innerModel = new SequenceModule<Matrix>
        {
            Inner = [
                new LinearMatrixModule(EMBEDDING_SIZE, HIDDEN_SIZE),
                LeakyReLUActivation.Instance,
                new LinearMatrixModule(HIDDEN_SIZE, HIDDEN_SIZE),
                LeakyReLUActivation.Instance,
                new LinearMatrixModule(HIDDEN_SIZE, HIDDEN_SIZE),
                LeakyReLUActivation.Instance,
                new LinearMatrixModule(HIDDEN_SIZE, HIDDEN_SIZE),
                LeakyReLUActivation.Instance,
                new LinearMatrixModule(HIDDEN_SIZE, HIDDEN_SIZE),
                LeakyReLUActivation.Instance,
                new LinearMatrixModule(HIDDEN_SIZE, EMBEDDING_SIZE),
                LeakyReLUActivation.Instance,
            ],
        };

        var input = new IndexEmbeddingModule(Tokenizer.TokenCount, EMBEDDING_SIZE);
        var output = new IndexUnembeddingModule(new(Tokenizer.TokenCount, weightedRandom: true, random), EMBEDDING_SIZE);

        NumericsInitializer.XavierUniform(input.EmbeddingMatrix, random);
        NumericsInitializer.XavierUniform(output.UnembeddingMatrix, random);

        var initer = new LinearModule.KaimingInitializer(LeakyReLUActivation.Instance) { Random = random };
        foreach (var linear in innerModel.Inner.OfType<LinearMatrixModule>())
        {
            initer.Init(linear);
        }

        return new EmbeddedModule<int[], Matrix, int>
        {
            Input = input,
            Hidden = innerModel,
            Output = output,
        };
    }

    public static void Run(ThreadingMode threading, Random random)
    {
        var trainingConfig = new TrainingConfig
        {
            EpochCount = 2,
            Optimizer = new AdamOptimizer
            {
                LearningRate = 0.001f,
            },

            EvaluationCallbackAfterBatches = 8,
            EvaluationCallback = evaluation => Console.WriteLine(evaluation),
            Threading = threading, // half seems to be faster than full
        };

        // var model = ModuleSerializer.Read<EmbeddedModule<int[], Matrix, int>>(ModelFile);
        var model = CreateAndInitModel(random);

        using var dataSet = new C4DataSource(initalFile: 0);
        // var dataSet = AssetManager.Sentences;
        var trainingSource = GetTrainingSource(dataSet, random);

        var trainer = new EmbeddedModuleTrainer<int[], Matrix, int>(model, trainingConfig)
        {
            CostFunction = CrossEntropyCostFromLogits.Instance,
            TrainingData = trainingSource,
        };

        ((IndexUnembeddingModule)model.Output).OuputLogits = true;
        trainer.TrainConsole();
        // Console.WriteLine(dataSet.GetState());

        trainer.DataPool.Clear();

        // ModuleSerializer.Write(model, ModelFile);

        ((IndexUnembeddingModule)model.Output).OuputLogits = false;
        LMHelper.StartChat(model, CONTEXT_SIZE, Tokenizer);
    }

    public static MemoryTrainingDataSource<TrainingEntry> GetTrainingSource(FileInfo fileInfo, Random random)
    {
        return new(Prepare(LanguageDataHelper.GetLinesPrintStatsToConsole(fileInfo), stride: 8)) { BatchCount = 64, Random = random };
    }

    public static SequenceTrainingDataSource<TrainingEntry> GetTrainingSource(C4DataSource dataSource, Random? random = null)
    {
        return new(Prepare(dataSource.GetLines(), stride: 92)) { BatchSize = 512 };
    }

    public static IEnumerable<TrainingEntry> Prepare(IEnumerable<string> data, int stride)
    {
        return data.TokenizeSkipInvalid(Tokenizer)
            .SlidingWindow(Tokenizer.TokenizeSingle("\0"), CONTEXT_SIZE, stride)
            .ToTrainingDataMatrix(Tokenizer.TokenCount);
    }
}
