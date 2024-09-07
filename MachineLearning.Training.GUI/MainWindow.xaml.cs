using System.Globalization;
using System.Windows;
using MachineLearning.Model;
using MachineLearning.Model.Layer;
using MachineLearning.Samples;
using MachineLearning.Samples.MNIST;
using MachineLearning.Serialization;
using MachineLearning.Serialization.Activation;
using MachineLearning.Training.Cost;
using MachineLearning.Training.Optimization.Adam;
using MachineLearning.Training.Optimization.SGDMomentum;
using SkiaSharp;

namespace MachineLearning.Training.GUI;

/// <summary>
/// Interaction logic for MainWindow.xaml
/// </summary>
public partial class MainWindow : Window
{
    public TrainingProgressTracker ProgressTracker { get; } = new();


    public MainWindow()
    {
        InitializeComponent();
        DataContext = this;

        CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;
        ActivationMethodSerializer.RegisterDefaults();

        //var serializer = new ModelSerializer(AssetManager.GetModelFile("mnist.nnw"));
        var model1 = MNISTModel.CreateModel(new Random(42));
        var model2 = MNISTModel.CreateModel(new Random(42));
        //var model = serializer.Load(MNISTEmbedder.Instance).ReduceOrThrow();
        var config = MNISTModel.GetTrainingConfig();

        var trainer1 = ProgressTracker.CreateLinkedTrainer("Adam Optimizer", SKColors.Blue, model1, config with { Optimizer = new AdamOptimizer
        {
            LearningRate = 0.1,
            CostFunction = CrossEntropyLoss.Instance,
        },
        DumpEvaluationAfterBatches = 4,
        });
        
        var trainer2 = ProgressTracker.CreateLinkedTrainer("SGD Optimizer", SKColors.Red, model2, config with { Optimizer = new GDMomentumOptimizer
        {
            InitialLearningRate = 0.7,
            CostFunction = CrossEntropyLoss.Instance,
        },
        DumpEvaluationAfterBatches = 4,

        });

        Loaded += (sender, args) =>
        {
            StatusLabel.Content = "Training...";
            _ = Task.Run(() => trainer1.Train());
            _ = Task.Run(() => trainer2.Train());
            //serializer.Save(model);
            //StatusLabel.Content = "Done!";
            //Task.Run(trainerTiny.Train);
        };
    }
}