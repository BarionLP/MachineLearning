using ML.Core.Evaluation.Cost;
using ML.Core.Modules;
using ML.Core.Modules.Activations;
using ML.Core.Modules.Builder;
using ML.Core.Training;
using ML.Core.Training.Data;
using ML.Runner;
using ML.Runner.Mnist;

// var random = Random.Shared;
var random = new Random(69);
var images = new MnistDataSource(AssetManager.MNISTArchive);

// TODO: some system to simplify initializer creation
var model = MultiLayerPerceptronBuilder.Create(784)
    .AddLayer(256, LeakyReLUActivation.Instance)
    .AddLayer(128, LeakyReLUActivation.Instance)
    .AddLayer(10, EmptyModule.Instance)  // only when training with CrossEntropyCostFromLogits
    // .AddLayer(10, (_, o) => new SoftMaxActivation(o)) // only when not training
    .Build();

var initializer = new SequenceModule<Vector>.Initializer
{
    Inner = [
        new PerceptronModule.KaimingInitializer(((PerceptronModule)model.Inner[0]).Activation),
        new PerceptronModule.KaimingInitializer(((PerceptronModule)model.Inner[1]).Activation),
        PerceptronModule.XavierInitializer.Instance,
        // EmptyModuleInitializer.Instance,
    ],
};

initializer.Init(model);

var embeddedModel = new EmbeddedModule<double[], Vector, int>
{
    Input = MnistInput.Instance,
    Hidden = model,
    Output = MnistOuput.Instance,
};

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
    RandomSource = random,
};

var trainer = new EmbeddedModuleTrainer<double[], Vector, int>(embeddedModel, trainingConfig)
{
    CostFunction = CrossEntropyCostFromLogits.Instance,
    TrainingData = new MnistDataSet(images.TrainingSet)
    {
        BatchCount = 128,
        Noise = new ImageNoise
        {
            Size = MnistImage.SIZE,
            NoiseStrength = 0.35,
            NoiseProbability = 0.75,
            MaxShift = 2,
            MaxAngle = 30,
            MinScale = 0.8,
            MaxScale = 1.2,
            Random = random,
        },
        Random = random,
    },
};

trainer.TrainConsole();

trainer.DataPool.Clear();


#if DEBUG
// forces all remaining finalizers to be called
GC.Collect();
#endif