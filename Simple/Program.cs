using Simple;
using Simple.Network;
using Simple.Network.Activation;
using Simple.Network.Embedding;
using Simple.Network.Layer;
using Simple.Serialization.Activation;
using Simple.Training;
using Simple.Training.Cost;
using Simple.Training.Data;

ActivationMethodSerializer.RegisterDefaults();

//var mnistDataSource = new MNISTDataSource(new(@"I:\Coding\TestEnvironments\NeuralNetwork\MNIST_ORG.zip"));
var mnistDataSource = new MNISTDataSource(new(@"C:\Users\Nation\OneDrive - Schulen Stadt Schwäbisch Gmünd\Data\MNIST_ORG.zip"));
var images = new ImageDataSource(new(@"C:\Users\Nation\OneDrive\Digits"));

var inputNoise = new ImageInputNoise {
    Size = MNISTDataPoint.SIZE,
    NoiseStrength = 0.3,
    NoiseProbability = 0.5,
    MaxShift = 2,
    MaxAngle = 35,
    MinScale = 0.8,
    MaxScale = 1.3,
    Random = new Random(3),
};

var config = new SimpleTrainingConfig<Number[], int>() {
    TrainingSet = mnistDataSource.TrainingSet,
    TestSet = mnistDataSource.TestingSet,
    LearnRate = 0.5,
    LearnRateMultiplier = 0.995,
    Regularization = 0.01,
    Momentum = 0.85,
    TrainingBatchSize = 512,
    TestBatchSize = 256,
    Iterations = 128*8,
    InputNoise = inputNoise,
    DumpEvaluationAfterIterations = 128,
    CostFunction = CrossEntropyCost.Instance,
    OutputResolver = new MNISTOutputResolver(),
    RandomSource = new Random(42),
};

// var count = 0;
// foreach(var item in config.GetNextTestBatch().Select(d=> new MNISTDataPoint(d.Input, d.Expected))){
//     item.SaveImage(new(@$"C:\Users\Nation\Downloads\digits\{count}.png"));
//     count++;
// }
// return;

var serializer = new NetworkSerializer<Number[], int, RecordingLayer>(new FileInfo(@"C:\Users\Nation\Downloads\digits.nnw"));
var setupRandom = new Random(69);

//var network = serializer.Load<RecordingNetwork<Number[], int>>(MNISTEmbedder.Instance).ReduceOrThrow();

var network = NetworkBuilder.Recorded<Number[], int>(784)
    .SetDefaultActivationMethod(LeakyReLUActivation.Instance)
    .SetEmbedder(MNISTEmbedder.Instance)
    .AddRandomizedLayer(128, setupRandom)
    .AddLayer(10, builder => builder.InitializeRandom(setupRandom).SetActivationMethod(SoftmaxActivation.Instance))
    .Build();

var trainer = new NetworkTrainer<Number[], int>(config, network);

var trainingResults = trainer.Train();
Console.WriteLine(trainingResults.DumpShort());

serializer.Save(network);

//Console.WriteLine(trainer.Evaluate().DumpShort());

foreach(var image in images.DataSet) {
    //Console.WriteLine();
    //Console.WriteLine(image.DumpImage());
    Console.Write($"Prediction: {network.Process(image.Image)} ");
    Console.WriteLine($"\t Actual: {image.Digit}");
}
