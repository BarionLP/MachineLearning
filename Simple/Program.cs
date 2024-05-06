using MachineLearning.Data.Entry;
using MachineLearning.Data.Noise;
using MachineLearning.Data.Source;
using MachineLearning.Domain.Activation;
using MachineLearning.Model;
using MachineLearning.Model.Embedding;
using MachineLearning.Serialization.Activation;
using MachineLearning.Training;
using MachineLearning.Training.Cost;
using MachineLearning.Training.Optimization;

ActivationMethodSerializer.RegisterDefaults();

//var mnistDataSource = new MNISTDataSource(new(@"I:\Coding\TestEnvironments\NeuralNetwork\MNIST_ORG.zip"));
var mnistDataSource = new MNISTDataSource(new(@"C:\Users\Nation\OneDrive - Schulen Stadt Schwäbisch Gmünd\Data\MNIST_ORG.zip"));
var images = new ImageDataSource(new(@"C:\Users\Nation\OneDrive\Digits"));

var inputNoise = new ImageInputNoise 
{
    Size = ImageDataEntry.SIZE,
    NoiseStrength = 0.35,
    NoiseProbability = 0.75,
    MaxShift = 2,
    MaxAngle = 30,
    MinScale = 0.8,
    MaxScale = 1.2,
    Random = new Random(3),
};

var config = new TrainingConfig<Number[], int>() 
{
    TrainingSet = mnistDataSource.TrainingSet,
    TestSet = mnistDataSource.TestingSet,
    
    EpochCount = 4,
    BatchSize = 256*2,

    /*
    Optimizer = new GDMomentumOptimizer{
        InitalLearningRate = 0.7,
        LearningRateEpochMultiplier = 0.5,
        Momentum = 0.85,
        Regularization = 0.01,
    },
    */

    Optimizer = new AdamOptimizerConfig 
    {
        LearningRate = 0.1,
        CostFunction = MeanSquaredErrorCost.Instance,
    },
    
    InputNoise = inputNoise,
    OutputResolver = new MNISTOutputResolver(),
    
    EvaluationCallback = result => Console.WriteLine(result.Dump()),
    DumpEvaluationAfterBatches = 4,
    
    RandomSource = new Random(42),
};

/*
using var serializer = new NetworkSerializer<Number[], int, RecordingLayer>(new FileInfo(@"C:\Users\Nation\Downloads\digits_big.nnw"));

var count = 0;
foreach(var item in config.GetRandomTestBatch().ApplyNoise(inputNoise).Select(d=> new MNISTDataPoint(d.Input, d.Expected))){
    item.SaveImage(new(@$"C:\Users\Nation\Downloads\digits\{count}.png"));
    count++;
}
return;

*/
var setupRandom = new Random(69);
var network = NetworkBuilder.Recorded<Number[], int>(784)
    .SetDefaultActivationMethod(LeakyReLUActivation.Instance)
    .SetEmbedder(MNISTEmbedder.Instance)
    .AddRandomizedLayer(256, setupRandom)
    .AddRandomizedLayer(128, setupRandom)
    .AddLayer(10, builder => builder.InitializeRandom(setupRandom).SetActivationMethod(SoftmaxActivation.Instance))
    .Build();

//var network = serializer.Load<RecordingNetwork<Number[], int>>(MNISTEmbedder.Instance).ReduceOrThrow();
var trainer = new NetworkTrainer<Number[], int>(config, network);

var trainingResults = trainer.Train();
Console.WriteLine(trainingResults.DumpShort());

//serializer.Save(network);

//Console.WriteLine(trainer.Evaluate().DumpShort());

var correctCounter  = 0;
var counter = 0;
var previousColor = Console.ForegroundColor;
foreach(var image in images.DataSet) 
{
    var prediction = network.Process(image.Image);
    if(prediction == image.Digit) correctCounter++;

    Console.ForegroundColor = prediction == image.Digit ? ConsoleColor.Green : ConsoleColor.Red;
    Console.WriteLine($"Predicted: {prediction}\tActual: {image.Digit}");
    counter++;
}
Console.ForegroundColor = previousColor;
Console.WriteLine($"Correct: {(double)correctCounter/counter:P0}");


/*
When scaling up neural network models in size and complexity, various hyperparameters need adjustment to maintain or improve the model’s training efficiency and performance. Here's a table overview that outlines general trends for tweaking key hyperparameters like Epoch Count, Batch Size, Learning Rate, Learning Rate Multiplier, Momentum, and Regularization as the model size increases:

| Hyperparameter        | Adjustment for Larger Model | Rationale                                                      |
|-----------------------|-----------------------------|----------------------------------------------------------------|
| **Epoch Count**       | Increase                    | Larger networks often require more training to converge as they can capture more complex patterns and may need more iterations through the data to adequately fit all parameters. |
| **Batch Size**        | Increase                    | Larger batches can provide more stable gradient estimates, which is beneficial for training larger networks. However, memory constraints and the point of diminishing returns on the hardware efficiency need consideration. |
| **Learning Rate**     | Decrease                    | A lower learning rate can help prevent overshooting the minimum during optimization. Larger models are more susceptible to destabilizing due to larger updates. |
| **Learning Rate Multiplier** | Adjust based on layer or parameter sensitivity | In larger models, finer control of learning rates across different layers can help address the varying learning speed of features, often decreasing the learning rate more on deeper layers to avoid instabilities. |
| **Momentum**          | Adjust as needed            | While momentum helps accelerate convergence in the relevant direction and dampens oscillations, the optimal setting might need tuning based on the network's response to updates, especially if training becomes unstable. |
| **Regularization**    | Increase                    | Larger models are more prone to overfitting due to their increased capacity. Regularization (e.g., L2, dropout) helps mitigate this by penalizing large weights or randomly dropping units during training. |

### Explanation of Adjustments:

- **Epoch Count:**More parameters and more complex functions necessitate longer training to explore the loss landscape adequately.
  
- **Batch Size:**Larger models benefit from larger batch sizes because they provide a more accurate estimate of the gradient. However, the ideal batch size should balance between computational efficiency (larger batches can be processed faster on parallel architectures like GPUs) and training stability.

- **Learning Rate:**Smaller learning rates help in fine-tuning the adjustments in weights without causing significant disruptions in the learned patterns, which is critical as the model size increases and the surface of the optimization landscape becomes more complex.

- **Learning Rate Multiplier:**This allows different parts of the network to train at different speeds, which can be particularly useful in very deep networks where earlier layers might need less adjustment as training progresses.

- **Momentum:**Maintaining or adjusting momentum is crucial since it helps in overcoming local minima and accelerates convergence, but too much momentum can cause overshooting in larger models where the gradients are inherently more variable.

- **Regularization:**As the capacity to memorize data increases with model size, regularization becomes more important to ensure that the model generalizes well to unseen data instead of memorizing the training set.

These adjustments are general guidelines and should be tailored to specific models and training conditions through systematic hyperparameter tuning, such as using grid search or Bayesian optimization methods.
*/