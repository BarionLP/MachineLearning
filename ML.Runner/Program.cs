using ML.Core.Evaluation;
using ML.Core.Modules;
using ML.Core.Modules.Activations;
using ML.Core.Training;
using ML.Core.Training.Data;
using ML.Runner;

var random = new Random(69);
// var random = new Random(69);
var images = new MnistDataSource(AssetManager.MNISTArchive);

var model = new SequenceModule<Vector>
{
    Inner = [
        new PerceptronModule(784, 256) { Activation = new LeakyReLUActivation(256) },
        new PerceptronModule(256, 128) { Activation = new LeakyReLUActivation(128) },
        new PerceptronModule(128, 10) { Activation = new SoftMaxActivation(10) },
    ],
};

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
    Threading = ThreadingMode.Single,
    RandomSource = random,
};

var trainer = new EmbeddedModuleTrainer<double[], Vector, int>(embeddedModel, trainingConfig)
{
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
    CostFunction = CrossEntropyLoss.Instance,
};

trainer.TrainConsole();
