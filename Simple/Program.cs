using Simple;
using Simple.Network;
using Simple.Network.Activation;
using Simple.Network.Embedding;
using Simple.Network.Layer;
using Simple.Training;
using Simple.Training.Cost;
using Simple.Training.Data;

var mnistDataSource = new MNISTDataSource(new(@"I:\Coding\TestEnvironments\NeuralNetwork\MNIST_ORG.zip"));

var config = new TrainingConfig<Number[], int>() {
    TrainingSet = mnistDataSource.TrainingSet,
    TestSet = mnistDataSource.TestingSet.Take(512).ToArray(),
    LearnRate = .25,
    LearnRateMultiplier = 0.999,
    BatchSize = 256,
    Iterations = 1028 * (2),
    DumpEvaluationAfterIterations = 32,
    CostFunction = CrossEntropyCost.Instance,
    OutputResolver = new MNISTOutputResolver(),
};

var network = new NetworkBuilder<RecordingNetwork<Number[], int>, Number[], int, RecordingLayer>(784)
    .SetEmbedder(MNISTEmbedder.Instance)
    .SetDefaultActivationMethod(LeakyReLUActivation.Instance)
    .AddRandomizedLayer(128)
    .AddRandomizedLayer(10)
    .Build();

//var trainer = new NetworkTrainer<Number[], int>(config, network);

//trainer.Train();

var serializer = new NetworkSerializer<Number[], int, RecordingLayer>(new FileInfo(@"C:\Users\Barion\Downloads\test.nnw"));
serializer.Save(network);

var deserializer = new NetworkSerializer<Number[], int, SimpleLayer>(new FileInfo(@"C:\Users\Barion\Downloads\test.nnw"));
var newNet = deserializer.Load<SimpleNetwork<Number[], int, SimpleLayer>>(network.ActivationMethod);

return;

//var network = new RecordingNetwork(2, 6, 4, 2) { ActivationMethod = SigmoidActivation.Instance };

//var config = new TrainingConfig() {
//    TrainingSet = ConstructTrainingData(1028*2).ToArray(),
//    TestSet = ConstructTrainingData(512).ToArray(),
//    LearnRate = .25,
//    LearnRateMultiplier = 0.99997,
//    BatchSize = 128,
//    Iterations = 1028*(64+16),
//    DumpEvaluationAfterIterations = 1028*2,
//    CostFunction = CrossEntropyCost.Instance,
//};


//var trainer = new NetworkTrainer(config, network);

//var viewSize = 48;

////WriteActualView(viewSize);
////return;

//Console.WriteLine("Starting Training...");

//var trainingResult = trainer.Train();


//Console.WriteLine("After Training:");
//Console.WriteLine($"Average Costs: {trainingResult.After.TrainingSetResult.AverageCost} | {trainingResult.After.TestSetResult.AverageCost}");
//Console.WriteLine($"Correct: {trainingResult.After.TrainingSetResult.CorrectPercentage:P} | {trainingResult.After.TestSetResult.CorrectPercentage:P}");

//Console.WriteLine();
//Console.WriteLine("Trained Model:");
//WriteModelView(viewSize);
//Console.WriteLine();
//Console.WriteLine("Actual:");
//WriteActualView(viewSize);
//Console.WriteLine();


//void WriteModelView(int size) {
//    foreach (var lineIndex in ..(size/2)) {
//        foreach(var charIndex in ..size) {
//            var result = network.Process([(Number) charIndex / size, (Number) lineIndex / (size / 2)]);
//            //Console.Write($"{result[0]*100:F0} ");
//            Console.Write(result[0] > result[1] ? '0' : ' ');
//        }
//        Console.WriteLine();
//    }
//}

//void WriteActualView(int size) {
//    foreach(var lineIndex in ..(size/2)) {
//        foreach(var charIndex in ..size) {
//            Console.Write(IsInsideShapes((Number) charIndex / size, (Number) lineIndex / (size/2)) ? '0' : ' ');
//        }
//        Console.WriteLine();
//    }
//}

//static bool IsInsideShapes(double x, double y) {
//    // Normalize inputs to the range [-1, 1] for both x and y
//    x = 2 * (x - 0.5);
//    y = 2 * (y - 0.5);

//    y = -y;

//    // Check if inside circle centered at (0, 0) with radius 0.5
//    bool insideCircle = Math.Pow(x, 2) + Math.Pow(y, 2) <= Math.Pow(0.5, 2);

//    // Check if inside rectangle from (-0.5, -0.5) to (0.5, 0.5)
//    bool insideRectangle = (x >= -1.0 && x <= 0.5) && (y >= -0.0 && y <= 0.5);

//    return insideCircle || insideRectangle;
//}


//IEnumerable<DataPoint> ConstructTrainingData(int count) {
//    foreach(var _ in ..count) {
//        var x = Random.Shared.NextDouble();
//        var y = Random.Shared.NextDouble();
//        yield return new DataPoint([x, y],  IsInsideShapes(x, y) ? [1, 0] : [0, 1]);
//    }
//}
