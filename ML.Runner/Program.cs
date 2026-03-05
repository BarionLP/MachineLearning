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

// TODO: build together with initer, allow auto init or custom init, when done remove EmptyModule case from PerceptronModule.Initializer
var model = MultiLayerPerceptronBuilder.Create(784)
    .AddLayer(256, (_, o) => new LeakyReLUActivation(o))
    .AddLayer(128, (_, o) => new LeakyReLUActivation(o))
    .AddLayer(10, (_, o) => new EmptyModule())
    .Build();

var initializer = new SequenceModule<Vector>.SharedInitializer
{
    Inner = new PerceptronModule.Initializer { Random = random },
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
