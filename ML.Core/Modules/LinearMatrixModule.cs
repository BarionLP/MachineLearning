using ML.Core.Attributes;

namespace ML.Core.Modules;

[GeneratedModule(IncludeSerializer: true)]
public sealed partial class LinearMatrixModule(Matrix weights, Vector biases) : IHiddenModule<Matrix>
{
    [Weights] public Matrix Weights { get; } = weights;
    [Weights] public Vector Biases { get; } = biases;

    public int InputColumnCount => Weights.RowCount;
    public int OutputColumnCount => Weights.ColumnCount;

    public LinearMatrixModule(int inputColumnCount, int outputColumnCount)
        : this(Matrix.Create(inputColumnCount, outputColumnCount), Vector.Create(outputColumnCount)) { }

    public Matrix Forward(Matrix input, Snapshot snapshot)
    {
        Debug.Assert(input.ColumnCount == InputColumnCount);

        snapshot.Input = input;

        input.MultiplyTo(Weights, snapshot.Weighted);
        foreach (var rowIndex in ..snapshot.Weighted.RowCount)
        {
            snapshot.Weighted.RowRef(rowIndex).AddTo(Biases, snapshot.Biased.RowRef(rowIndex));
        }
        return snapshot.Biased;
    }

    public Matrix Backward(Matrix outputGradient, Snapshot snapshot, Gradients gradients)
    {
        foreach (var rowIndex in ..outputGradient.RowCount)
        {
            gradients.Biases.AddToSelf(outputGradient.RowRef(rowIndex));
        }

        snapshot.Input.TransposeLeftMultiplyAddTo(outputGradient, gradients.Weights);
        outputGradient.TransposeRightMultiplyTo(Weights, snapshot.InputGradient);
        return snapshot.InputGradient;
    }

    partial class Snapshot
    {
        public Matrix Input
        {
            get;
            set
            {
                field = value;

                WeightedStorage.SetCount(field.RowCount, module.OutputColumnCount);
                BiasedStorage.SetCount(field.RowCount, module.OutputColumnCount);
                InputGradientStorage.OfSize(field);
            }
        }

        public Matrix Weighted => WeightedStorage.Tensor;
        public Matrix Biased => BiasedStorage.Tensor;
        public Matrix InputGradient => InputGradientStorage.Tensor;

        private readonly Dynamic<Matrix> WeightedStorage = new();
        private readonly Dynamic<Matrix> BiasedStorage = new();
        private readonly Dynamic<Matrix> InputGradientStorage = new();
    }

    [GeneratedAdam(typeof(LinearMatrixModule))]
    public sealed partial class Adam;
}