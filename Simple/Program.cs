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

//Console.WriteLine(images.DataSet[0].DumpImage());

//return;
var config = new SimpleTrainingConfig<Number[], int>() {
    TrainingSet = mnistDataSource.TrainingSet,
    TestSet = mnistDataSource.TestingSet,
    LearnRate = 0.7,
    LearnRateMultiplier = 0.994,
    TrainingBatchSize = 256,
    TestBatchSize = 256,
    Iterations = 128*4,
    InputNoise = new RandomInputNoise(0.2f, new Random(420)),
    DumpEvaluationAfterIterations = 48,
    CostFunction = CrossEntropyCost.Instance,
    OutputResolver = new MNISTOutputResolver(),
    RandomSource = new Random(42),
};

var serializer = new NetworkSerializer<Number[], int, RecordingLayer>(new FileInfo(@"C:\Users\Nation\Downloads\digits.nnw"));
var setupRandom = new Random(69);

//var network = serializer.Load<RecordingNetwork<Number[], int>>(MNISTEmbedder.Instance).ReduceOrThrow();

var network = NetworkBuilder.Recorded<Number[], int>(784)
    .SetDefaultActivationMethod(LeakyReLUActivation.Instance)
    .SetEmbedder(MNISTEmbedder.Instance)
    //.AddRandomizedLayer(128, setupRandom)
    .AddRandomizedLayer(128, setupRandom)
    .AddLayer(10, builder => builder.InitializeRandom(setupRandom).SetActivationMethod(SoftmaxActivation.Instance))
    .Build();

var trainer = new NetworkTrainer<Number[], int>(config, network);

var trainingResults = trainer.Train();
Console.WriteLine(trainingResults.DumpShort());

serializer.Save(network);

//Console.WriteLine(trainer.Evaluate().DumpShort());

foreach(var image in images.DataSet) {
    Console.WriteLine();
    Console.WriteLine(image.DumpImage());
    Console.Write($"Prediction: {network.Process(image.Image)} ");
    Console.WriteLine($"\t\t Actual: {image.Digit}");
}
