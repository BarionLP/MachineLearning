using System.Diagnostics;
using MachineLearning.Model.Activation;
using MachineLearning.Model.Attributes;
using MachineLearning.Model.Initialization;
using MachineLearning.Model.Layer;
using MachineLearning.Model.Layer.Initialization;
using MachineLearning.Serialization;
using MachineLearning.Training.Attributes;
using MachineLearning.Training.Cost;
using MachineLearning.Training.Optimization;
using MachineLearning.Training.Optimization.Adam;
using static MachineLearning.Serialization.ModelSerializationHelper;

namespace MachineLearning.Mamba;

[GeneratedLayer, LayerSerializer("unemb", 2), GenerateOptimizers(OutputGradientType = typeof(Matrix))]
public sealed partial class UnEmbeddingLayer : ILayer<Matrix, (Vector, int), UnEmbeddingLayer.Snapshot>
{
    [Parameter] public int ContextSize { get; }
    // init randomly with [-0.1; 0.1] or [-0.01; 0.01]
    [Weights] public Matrix UnEmbeddingMatrix { get; }
    public int TokenCount => UnEmbeddingMatrix.RowCount;

    public int EmbeddingSize => UnEmbeddingMatrix.ColumnCount;

    public UnEmbeddingLayer(int tokenCount, int contextSize, int embeddingSize)
        : this(contextSize, Matrix.Create(tokenCount, embeddingSize)) { }

    // TODO: incorporate all token predictions in the gradient
    public (Matrix, int) Forward(Matrix input, Snapshot snapshot)
    {
        Debug.Assert(input.RowCount == ContextSize);
        Debug.Assert(input.ColumnCount == EmbeddingSize);

        input.CopyTo(snapshot.Input);

        foreach (var i in ..input.RowCount)
        {
            UnEmbeddingMatrix.MultiplyTo(input.RowRef(i), snapshot.WeightedInput);
            SoftMaxActivation.Instance.ActivateTo(snapshot.WeightedInput, snapshot.Output.RowRef(i));
        }

        return (snapshot.Output, snapshot.Output.RowRef(snapshot.Output.RowCount - 1).MaximumIndex());
    }

    public void Backward(Matrix outputGradients, Snapshot snapshot)
    {
        Debug.Assert(outputGradients.ColumnCount == TokenCount);
        Debug.Assert(outputGradients.RowCount == snapshot.Input.RowCount);

        snapshot.GradientUnEmbeddingMatrix.ResetZero();

        // this would be neccecary without CrossEntropyFromSoftmaxLoss;
        // var tmp = Vector.Create(outputGradient.Count);
        // SoftMaxActivation.Instance.DerivativeTo(snapshot.WeightedInput, tmp);
        // tmp.PointwiseMultiplyToSelf(outputGradient);

        // y = W * v
        // dy = y - expected // because CrossEntropy and Softmax cancel out
        // => dW += v * dy
        // => dv += W^T * dy

        foreach (var i in ..outputGradients.RowCount)
        {
            VectorHelper.MultiplyToMatrixAddTo(outputGradients.RowRef(i), snapshot.Input.RowRef(i), snapshot.GradientUnEmbeddingMatrix);
            UnEmbeddingMatrix.MultiplyTransposedTo(outputGradients.RowRef(i), snapshot.InputGradient.RowRef(i));
        }

        // snapshot.GradientUnEmbeddingMatrix.DivideToSelf(outputGradients.RowCount);
    }


    public static Result<UnEmbeddingLayer> Readv1(BinaryReader reader)
    {
        var tokenCount = reader.ReadInt32();
        var contextSize = reader.ReadInt32();
        var embeddingSize = reader.ReadInt32();
        var matrix = ReadMatrixRaw(tokenCount, embeddingSize, reader);
        return new UnEmbeddingLayer(contextSize, matrix);
    }

    partial class Snapshot
    {
        public Matrix Input { get; } = Matrix.Create(layer.ContextSize, layer.EmbeddingSize);
        public Vector WeightedInput { get; } = Vector.Create(layer.TokenCount);
        public Matrix Output { get; } = Matrix.Create(layer.ContextSize, layer.TokenCount);

        public Matrix InputGradient { get; } = Matrix.Create(layer.ContextSize, layer.EmbeddingSize);
    }

    public sealed class Initializer(Random? random = null) : IInitializer<UnEmbeddingLayer>
    {
        public Random Random { get; } = random ?? Random.Shared;

        public void Initialize(UnEmbeddingLayer layer)
        {
            layer.UnEmbeddingMatrix.MapToSelf(_ => InitializationHelper.RandomInNormalDistribution(Random, 0, 0.03f));
        }
    }
}

// public sealed class UnEmbeddingLayerAdam : ILayerOptimizer<UnEmbeddingLayer, UnEmbeddingLayer.Snapshot>
// {
//     public UnEmbeddingLayer Layer { get; }
//     public ICostFunction CostFunction => Optimizer.CostFunction;
//     public AdamOptimizer Optimizer { get; }

//     public Matrix Gradient { get; }

//     // formula symbol M 
//     // exponentially decaying average of past gradients. It is akin to the mean of the gradients.
//     public readonly Matrix FirstMoment;

//     // formula symbol V
//     // exponentially decaying average of the squared gradients. It is akin to the uncentered variance of the gradients.
//     public readonly Matrix SecondMoment;


//     public UnEmbeddingLayerAdam(AdamOptimizer optimizer, UnEmbeddingLayer layer)
//     {
//         Optimizer = optimizer;
//         Layer = layer;

//         Gradient = Matrix.OfSize(Layer.UnEmbeddingMatrix);
//         FirstMoment = Matrix.OfSize(Layer.UnEmbeddingMatrix);
//         SecondMoment = Matrix.OfSize(Layer.UnEmbeddingMatrix);
//     }

//     private readonly Lock _lock = new();
//     public void Update(Vector nodeValues, UnEmbeddingLayer.Snapshot snapshot)
//     {
//         NumericsDebug.AssertValidNumbers(nodeValues);
//         Layer.Backward(nodeValues, snapshot);

//         NumericsDebug.AssertValidNumbers(snapshot.InputGradient);
//         NumericsDebug.AssertValidNumbers(snapshot.GradientUnEmbeddingMatrix);

//         lock (_lock)
//         {
//             Gradient.AddToSelf(snapshot.GradientUnEmbeddingMatrix);
//         }
//     }

//     public void Apply(int dataCounter)
//     {
//         (FirstMoment, Gradient).MapToFirst(FirstMomentEstimate);
//         (SecondMoment, Gradient).MapToFirst(SecondMomentEstimate);
//         Layer.UnEmbeddingMatrix.SubtractToSelf((FirstMoment, SecondMoment).Map(WeightReduction));
//     }

//     private float WeightReduction(float firstMoment, float secondMoment)
//     {
//         var mHat = firstMoment / (1 - Weight.Pow(Optimizer.FirstDecayRate, Optimizer.Iteration));
//         var vHat = secondMoment / (1 - Weight.Pow(Optimizer.SecondDecayRate, Optimizer.Iteration));
//         return Optimizer.LearningRate * mHat / (Weight.Sqrt(vHat) + Optimizer.Epsilon);
//     }

//     private float FirstMomentEstimate(float lastMoment, float gradient)
//         => Optimizer.FirstDecayRate * lastMoment + (1 - Optimizer.FirstDecayRate) * gradient;
//     private float SecondMomentEstimate(float lastMoment, float gradient)
//         => Optimizer.SecondDecayRate * lastMoment + (1 - Optimizer.SecondDecayRate) * gradient * gradient;

//     public void GradientCostReset()
//     {
//         Gradient.ResetZero();
//     }

//     public void FullReset()
//     {
//         GradientCostReset();

//         FirstMoment.ResetZero();
//         SecondMoment.ResetZero();
//     }
// }