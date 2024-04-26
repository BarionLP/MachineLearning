using Simple.Network;
using Simple.Network.Activation;
using Simple.Network.Embedding;
using Simple.Training;
using Simple.Training.Cost;

namespace Simple;

public static class BinaryClassifier{
    public static void Code(){
        var network = NetworkBuilder.Recorded<Number[], bool>(2)
            .SetDefaultActivationMethod(SigmoidActivation.Instance)
            .SetEmbedder(new Embedder())
            .AddRandomizedLayer(6)
            .AddRandomizedLayer(4)
            .AddRandomizedLayer(2)
            .Build();

        var config = new TrainingConfig<Number[], bool>() {
           TrainingSet = ConstructTrainingData(1028*2).ToArray(),
           TestSet = ConstructTrainingData(512).ToArray(),
           EpochCount = 64+16,
           LearnRate = .25,
           LearnRateMultiplier = 0.99997,
           Regularization = 0,
           Momentum = 0,
           BatchSize = 128,
           DumpEvaluationAfterBatches = 128,
           OutputResolver = new OutputResolver(),
           CostFunction = CrossEntropyCost.Instance,
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


        static IEnumerable<BinaryDataPoint> ConstructTrainingData(int count) {
            foreach(var _ in ..count) {
                var x = Random.Shared.NextDouble();
                var y = Random.Shared.NextDouble();
                yield return new BinaryDataPoint([x, y],  IsInsideShapes(x, y));
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
