namespace MachineLearning.Samples;

public static class BinaryClassifier
{
    public static SimpleNetwork<double[], bool, RecordingLayer> GetModel()
    {
        var initializer = XavierInitializer.Instance;
        return NetworkBuilder.Recorded<double[], bool>(2)
            .SetDefaultActivationMethod(SigmoidActivation.Instance)
            .SetEmbedder(new Embedder())
            .AddLayer(7, initializer)
            .AddLayer(4, initializer)
            .AddLayer(2, initializer)
            .Build();
    }

    public static TrainingConfig<double[], bool> GetTrainingConfig()
    {
        return new TrainingConfig<double[], bool>()
        {
            TrainingSet = ConstructTrainingData(1028 * 12).ToArray(),
            TestSet = ConstructTrainingData(1028).ToArray(),

            EpochCount = 64 * 4,
            BatchCount = 32,

            Optimizer = new AdamOptimizerConfig
            {
                LearningRate = 0.2,
                CostFunction = CrossEntropyLoss.Instance,
            },

            OutputResolver = new OutputResolver(),

            ShuffleTrainingSetPerEpoch = true,

            EvaluationCallback = result => Console.WriteLine(result.Dump()),
        };
    }

    public static void TrainDefault()
    {

        var model = GetModel();
        var config = GetTrainingConfig();
        var trainer = new NetworkTrainer<double[], bool>(config, model);

        trainer.Train();

        var viewSize = 48;

        Console.WriteLine("Trained Model: ");
        WriteModelView(viewSize);
        Console.WriteLine();
        Console.WriteLine("Actual: ");
        WriteActualView(viewSize);

        void WriteModelView(int size)
        {
            foreach(var lineIndex in ..(size / 2))
            {
                foreach(var charIndex in ..size)
                {
                    var result = model.Process([(double) charIndex / size, (double) lineIndex / (size / 2)]);
                    //Console.Write($"{result[0]*100:F0} ");
                    Console.Write(result ? '0' : '.');
                }
                Console.WriteLine();
            }
        }

        static void WriteActualView(int size)
        {
            foreach(var lineIndex in ..(size / 2))
            {
                foreach(var charIndex in ..size)
                {
                    Console.Write(IsInsideShapes((double) charIndex / size, (double) lineIndex / (size / 2)) ? '0' : '.');
                }
                Console.WriteLine();
            }
        }
    }

    private static IEnumerable<BinaryDataEntry> ConstructTrainingData(int count)
    {
        foreach(var _ in ..count)
        {
            var x = Random.Shared.NextDouble();
            var y = Random.Shared.NextDouble();
            yield return new BinaryDataEntry([x, y], IsInsideShapes(x, y));
        }
    }

    private static bool IsInsideShapes(double x, double y)
    {
        x = 2 * (x - 0.5);
        y = 2 * (y - 0.5);

        y = -y;

        bool insideCircle = Math.Pow(x, 2) + Math.Pow(y, 2) <= Math.Pow(0.5, 2);
        bool insideRectangle = x >= -1.0 && x <= 0.5 && y >= -0.0 && y <= 0.5;

        return insideCircle || insideRectangle;
    }

    private sealed class OutputResolver : IOutputResolver<bool>
    {
        private static readonly Vector TRUE = Vector.Of([1, 0]);
        private static readonly Vector FALSE = Vector.Of([0, 1]);
        public Vector Expected(bool output) => output ? TRUE : FALSE;
    }
    public sealed class Embedder : IEmbedder<double[], bool>
    {
        public Vector Embed(double[] input) => Vector.Of(input);
        public bool UnEmbed(Vector input) => input[0] > input[1];
    }
}