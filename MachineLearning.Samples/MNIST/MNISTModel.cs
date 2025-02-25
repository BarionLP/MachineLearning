using MachineLearning.Data;
using MachineLearning.Data.Noise;
using MachineLearning.Data.Source;
using MachineLearning.Model.Layer;
using ML.MultiLayerPerceptron;
using ML.MultiLayerPerceptron.Initialization;
using ModelDefinition = ML.MultiLayerPerceptron.EmbeddedModel<double[], int>;

namespace MachineLearning.Samples.MNIST;

public static class MNISTModel
{
    public static IEmbeddingLayer<double[]> Embedder => MNISTEmbedder.Instance;
    public static IUnembeddingLayer<int> UnEmbedder => MNISTUnEmbedder.Instance;

    public static ModelDefinition CreateModel(Random? random = null)
    {
        var initializer = new HeInitializer(random);

        var network = AdvancedModelBuilder.Create(Embedder)
                .DefaultActivation(LeakyReLUActivation.Instance)
                .AddLayer(256, initializer)
                .AddLayer(128, initializer)
                .AddLayer(10, new XavierInitializer(random), SoftMaxActivation.Instance)
                .AddOutputLayer(UnEmbedder);

        return network;
    }

    public static ITrainingSet GetTrainingSet(Random? random = null)
    {
        var dataSource = new MNISTDataSource(AssetManager.MNISTArchive);

        return new MNISTDataSet(dataSource.TrainingSet)
        {
            BatchCount = 128,
            Noise = new ImageInputNoise
            {
                Size = ImageDataEntry.SIZE,
                NoiseStrength = 0.35,
                NoiseProbability = 0.75,
                MaxShift = 2,
                MaxAngle = 30,
                MinScale = 0.8,
                MaxScale = 1.2,
                Random = random ?? Random.Shared,
            },
            Random = random ?? Random.Shared,
        };
    }

    public static TrainingConfig DefaultTrainingConfig(Random? random = null)
    {
        return new TrainingConfig()
        {
            EpochCount = 8,

            Optimizer = new AdamOptimizer
            {
                LearningRate = 0.1f,
                CostFunction = CrossEntropyLoss.Instance,
            },

            DumpEvaluationAfterBatches = 8,
            EvaluationCallback = evaluation => Console.WriteLine(evaluation.Dump()),

            RandomSource = random ?? Random.Shared,

            MultiThread = false,
        };
    }

    public static ModelDefinition TrainDefault(ModelDefinition? model = null, TrainingConfig? config = null, Random? random = null)
    {
        model ??= CreateModel(random);
        var trainer = new EmbeddedModelTrainer<double[], int>(model, config ?? DefaultTrainingConfig(random), GetTrainingSet());

        trainer.TrainConsole();

        var images = new ImageDataSource(AssetManager.CustomDigits);
        Benchmark(model, images);

        return model;
    }

    public static void Benchmark(ModelDefinition model, ImageDataSource dataSource)
    {
        var correctCounter = 0;
        var counter = 0;
        var previousColor = Console.ForegroundColor;
        foreach (var image in dataSource.DataSet)
        {
            var (prediction, confidence) = model.Process(image.Image);

            if (prediction == image.Digit)
            {
                correctCounter++;
            }

            Console.ForegroundColor = prediction == image.Digit ? ConsoleColor.Green : ConsoleColor.Red;
            Console.WriteLine($"Predicted: {prediction} ({confidence:P})\tActual: {image.Digit}");
            counter++;
        }
        Console.ForegroundColor = previousColor;
        Console.WriteLine($"Correct: {(double)correctCounter / counter:P0}");
    }
}
