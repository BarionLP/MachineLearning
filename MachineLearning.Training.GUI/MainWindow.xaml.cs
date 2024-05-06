using System.Windows;
using MachineLearning.Data.Entry;
using MachineLearning.Data.Noise;
using MachineLearning.Data.Source;
using MachineLearning.Domain.Activation;
using MachineLearning.Model;
using MachineLearning.Model.Embedding;
using MachineLearning.Training.Cost;
using MachineLearning.Training.Optimization;
using SkiaSharp;

namespace MachineLearning.Training.GUI;

/// <summary>
/// Interaction logic for MainWindow.xaml
/// </summary>
public partial class MainWindow : Window {
    public TrainingProgressTracker ProgressTracker { get; } = new();
    

    public MainWindow() {
        InitializeComponent();
        DataContext = this;

        var networkBuilderHuge = NetworkBuilder.Recorded<double[], int>(784)
            .SetDefaultActivationMethod(LeakyReLUActivation.Instance)
            .SetEmbedder(MNISTEmbedder.Instance)
            .AddRandomizedLayer(256)
            .AddRandomizedLayer(128)
            .AddLayer(10, builder => builder.InitializeRandom().SetActivationMethod(SoftmaxActivation.Instance));
        
        var networkBuilderGPT = NetworkBuilder.Recorded<double[], int>(784)
            .SetDefaultActivationMethod(LeakyReLUActivation.Instance)
            .SetEmbedder(MNISTEmbedder.Instance)
            .AddRandomizedLayer(128)
            .AddLayer(10, builder => builder.InitializeRandom().SetActivationMethod(SoftmaxActivation.Instance));
        
        var mnistDataSource = new MNISTDataSource(new(@"C:\Users\Nation\OneDrive - Schulen Stadt Schwäbisch Gmünd\Data\MNIST_ORG.zip"));
        var images = new ImageDataSource(new(@"C:\Users\Nation\OneDrive\Digits"));

        var config = new TrainingConfig<double[], int>()
        {
            TrainingSet = mnistDataSource.TrainingSet,
            TestSet = mnistDataSource.TestingSet,

            EpochCount = 4,
            BatchSize = 256 * 2,

            Optimizer = new AdamOptimizerConfig 
            {
                LearningRate = 0.1,
                CostFunction = CrossEntropyLoss.Instance,
            },

            OutputResolver = new MNISTOutputResolver(),
            InputNoise = new ImageInputNoise
            {
                Size = ImageDataEntry.SIZE,
                NoiseStrength = 0.35,
                NoiseProbability = 0.75,
                MaxShift = 2,
                MaxAngle = 30,
                MinScale = 0.8,
                MaxScale = 1.2,
            },

            DumpEvaluationAfterBatches = 4,
        };

        var trainerBig = ProgressTracker.CreateLinkedTrainer("ChatGPT", SKColors.Blue, config, networkBuilderGPT.Build());
        var trainerSmall = ProgressTracker.CreateLinkedTrainer("Huge", SKColors.Red, config, networkBuilderHuge.Build());

        Loaded += (sender, args) => {
            Task.Run(trainerBig.Train);
            Task.Run(trainerSmall.Train);
            //Task.Run(trainerTiny.Train);
        };
    }
}