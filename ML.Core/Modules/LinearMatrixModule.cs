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

                WeightedStorage.SetCount(field.RowCount * module.OutputColumnCount);
                Weighted = Matrix.Of(field.RowCount, module.OutputColumnCount, WeightedStorage.Vector);

                BiasedStorage.SetCount(field.RowCount * module.OutputColumnCount);
                Biased = Matrix.Of(field.RowCount, module.OutputColumnCount, BiasedStorage.Vector);

                InputGradientStorage.SetCount(field.FlatCount);
                InputGradient = Matrix.OfSize(field, InputGradientStorage.Vector);
            }
        }

        public Matrix Weighted { get; private set; }
        public Matrix Biased { get; private set; }
        public Matrix InputGradient { get; private set; }

        private readonly DynamicVector WeightedStorage = new();
        private readonly DynamicVector BiasedStorage = new();
        private readonly DynamicVector InputGradientStorage = new();

        private void OnDispose()
        {
            Weighted = Matrix.Empty;
            Biased = Matrix.Empty;
            InputGradient = Matrix.Empty;
        }
    }

    [GeneratedAdam(typeof(LinearMatrixModule))]
    public sealed partial class Adam;
}