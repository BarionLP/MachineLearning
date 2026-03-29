using System.IO;
using ML.Core.Converters;
using ML.Core.Data.Noise;
using ML.Core.Data.Training;
using ML.Core.Evaluation.Cost;
using ML.Core.Modules;
using ML.Core.Modules.Activations;
using ML.Core.Modules.Builder;
using ML.Core.Training;

namespace ML.Runner.Samples.Mnist;

public static class MnistModel
{
    public static MnistDataSet DataSet => field ??= new MnistDataSet(AssetManager.MNISTArchive);
    public static int BatchCount => 128;
    public static FileInfo ModelFile { get; } = AssetManager.GetModelFile("mnist");

    public static SequenceModule<Vector> CreateAndInitModel(Random random) => MultiLayerPerceptronBuilder.Create(784)
        .AddLayer(256, LeakyReLUActivation.Instance)
        .AddLayer(128, LeakyReLUActivation.Instance)
        .AddLayer(10, SoftMaxActivation.Instance)
        .BuildAndInit(random);

    public static void Run(Random random)
    {
        var trainingConfig = new TrainingConfig
        {
            EpochCount = 1,
            Optimizer = new AdamOptimizer
            {
                LearningRate = 0.0046225016f,
            },

            EvaluationCallbackAfterBatches = 8,
            EvaluationCallback = evaluation => Console.WriteLine(evaluation),
            Threading = ThreadingMode.Full,
        };

        var model = ModuleSerializer.Read<SequenceModule<Vector>>(ModelFile);
        // var model = CreateAndInitModel(random);

        // modify the last layer to output logit instead of probabilites
        // so we can use the optimized version of CrossEntropyCost
        var last = (PerceptronModule)model.Inner[^1];
        var embeddedModel = new EmbeddedModule<double[], Vector, int>
        {
            Input = MnistInput.Instance,
            Hidden = new SequenceModule<Vector> { Inner = model.Inner.SetItem(model.Inner.Length - 1, new PerceptronModule(EmptyModule.Instance, last.Weights, last.Biases)) },
            Output = new IndexOutputLayer(tokenCount: 10, weightedRandom: false),
        };

        var trainer = new EmbeddedModuleTrainer<double[], Vector, int>(embeddedModel, trainingConfig)
        {
            CostFunction = CrossEntropyCostFromLogits.Instance,
            TrainingData = GetTrainingSource(random),
        };

        trainer.TrainConsole();

        // ModuleSerializer.Write(model, ModelFile);

        trainer.DataPool.Clear();

        var inferenceModel = new EmbeddedModule<double[], Vector, int>
        {
            Input = MnistInput.Instance,
            Hidden = model,
            Output = embeddedModel.Output,
        };

        Benchmark(inferenceModel, GetTestImages());
    }

    public static void Benchmark(IEmbeddedModule<double[], int> model, IEnumerable<(double[] Image, int Digit)> dataSource)
    {
        var correctCounter = 0;
        var counter = 0;
        var previousColor = Console.ForegroundColor;
        using var snapshot = model.CreateSnapshot();

        foreach (var (image, digit) in dataSource)
        {
            var (prediction, confidence) = model.Forward(image, snapshot);

            if (prediction == digit)
            {
                correctCounter++;
            }

            Console.ForegroundColor = prediction == digit ? ConsoleColor.Green : ConsoleColor.Red;
            Console.WriteLine($"Predicted: {prediction} ({confidence:P})\tActual: {digit}");
            counter++;
        }
        Console.ForegroundColor = previousColor;
        Console.WriteLine($"Correct: {(double)correctCounter / counter:P0}");
    }

    public static MnistImageSource GetTrainingSource(Random random) => GetDataSourceWithNoise(DataSet.TrainingSet, random);
    public static MnistImageSource GetDataSourceWithNoise(IEnumerable<MnistImage> images, Random random) => new(images)
    {
        BatchCount = BatchCount,
        Noise = new ImageNoise
        {
            Size = MnistImage.SIZE,
            NoiseStrength = 0.35,
            MaxShift = 2,
            MaxAngle = 30,
            MinScale = 0.8,
            MaxScale = 1.2,
            Random = random,
        },
        Random = random,
    };

    public static IEnumerable<(double[], int)> GetTestImages() 
        => AssetManager.CustomDigits.EnumerateFiles("*.png").Select(f => (ImageLoader.LoadGrayScale(f), f.NameWithoutExtension.Parse<int>()));
}
