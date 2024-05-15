using MachineLearning.Data.Entry;
using MachineLearning.Domain.Activation;
using MachineLearning.Model;
using MachineLearning.Model.Embedding;
using MachineLearning.Model.Layer.Initialization;
using MachineLearning.Training;
using MachineLearning.Training.Cost;
using MachineLearning.Training.Optimization;

namespace Simple;

public static class BinaryClassifier{
    public static void Code(){
        
        var initializer = new XavierInitializer();
        var network = NetworkBuilder.Recorded<Number[], bool>(2)
            .SetDefaultActivationMethod(SigmoidActivation.Instance)
            .SetEmbedder(new Embedder())
            .AddLayer(6, initializer)
            .AddLayer(4, initializer)
            .AddLayer(2, initializer)
            .Build();

        var config = new TrainingConfig<Number[], bool>()
        {
            TrainingSet = ConstructTrainingData(1028*2).ToArray(),
            TestSet = ConstructTrainingData(512).ToArray(),
            EpochCount = 64+16,

            Optimizer = new GDMomentumOptimizerConfig
            {
                LearningRate = .25,
                LearningRateEpochMultiplier = 0.99997,
                Regularization = 0,
                Momentum = 0,
                CostFunction = CrossEntropyLoss.Instance,
            },
            
            BatchSize = 128,
            DumpEvaluationAfterBatches = 128,
            OutputResolver = new OutputResolver(),
        };


        var trainer = new NetworkTrainer<Number[], bool>(config, network);

        var viewSize = 48;

        Console.WriteLine("Trained Model: ");
        WriteModelView(viewSize);
        Console.WriteLine();
        Console.WriteLine("Actual: ");
        WriteActualView(viewSize);

        void WriteModelView(int size) {
            foreach (var lineIndex in ..(size/2)) {
                foreach(var charIndex in ..size) {
                    var result = network.Process([(Number) charIndex / size, (Number) lineIndex / (size / 2)]);
                    //Console.Write($"{result[0]*100:F0} ");
                    Console.Write(result ? '0' : '.');
                }
                Console.WriteLine();
            }
        }

        static void WriteActualView(int size) {
            foreach(var lineIndex in ..(size/2)) {
                foreach(var charIndex in ..size) {
                    Console.Write(IsInsideShapes((Number) charIndex / size, (Number) lineIndex / (size/2)) ? '0' : ' ');
                }
                Console.WriteLine();
            }
        }

        static bool IsInsideShapes(double x, double y) {
            x = 2 * (x - 0.5);
            y = 2 * (y - 0.5);

            y = -y;

            bool insideCircle = Math.Pow(x, 2) + Math.Pow(y, 2) <= Math.Pow(0.5, 2);
            bool insideRectangle = (x >= -1.0 && x <= 0.5) && (y >= -0.0 && y <= 0.5);

            return insideCircle || insideRectangle;
        }


        static IEnumerable<BinaryDataEntry> ConstructTrainingData(int count) {
            foreach(var _ in ..count) {
                var x = Random.Shared.NextDouble();
                var y = Random.Shared.NextDouble();
                yield return new BinaryDataEntry([x, y],  IsInsideShapes(x, y));
            }
        }
    }

    private sealed class OutputResolver : IOutputResolver<bool, Number[]>{
        public Number[] Expected(bool output) => output ? [1, 0] : [0, 1];
    }
    public sealed class Embedder : IEmbedder<Number[], Number[], bool>{
        public Number[] Embed(Number[] input) => input;

        public bool UnEmbed(Number[] input) => input[0] > input[1];
    }
}
