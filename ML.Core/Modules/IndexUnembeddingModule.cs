using ML.Core.Attributes;

namespace ML.Core.Modules;

[GeneratedModule(IncludeSerializer: true)]
public sealed partial class IndexUnembeddingModule(IndexOutputModule output, Matrix unembeddingMatrix) : IOutputModule<Matrix, int>
{
    [SubModule] public IndexOutputModule Output { get; } = output;
    [Weights] public Matrix UnembeddingMatrix { get; } = unembeddingMatrix; // stored transposed to enable weight sharing with embedding module

    public int TokenCount => UnembeddingMatrix.RowCount;
    public int EmbeddingSize => UnembeddingMatrix.ColumnCount;

    public bool OuputLogits { get; set; } = false;

    public IndexUnembeddingModule(IndexOutputModule output, int embeddingSize)
        : this(output, Matrix.Create(output.TokenCount, embeddingSize)) { }

    public (int Output, Weight Confidence, Matrix Weights) Forward(Matrix input, Snapshot snapshot)
    {
        Debug.Assert(input.ColumnCount == EmbeddingSize);

        snapshot.Input = input;

        foreach (var i in ..input.RowCount)
        {
            UnembeddingMatrix.MultiplyTo(input.RowRef(i), snapshot.Unembedded.RowRef(i));
        }

        if (!OuputLogits)
        {
            foreach (var rowIndex in ..snapshot.Unembedded.RowCount)
            {
                snapshot.Unembedded.RowRef(rowIndex).SoftMaxToSelf();
            }
        }

        var (index, confidence, _) = Output.Forward(snapshot.Unembedded.RowRef(^1), snapshot.Output);
        return (index, confidence, snapshot.Unembedded);
    }

    public Matrix Backward(Matrix outputGradient, Snapshot snapshot, Gradients gradients)
    {
        Debug.Assert(outputGradient.ColumnCount == TokenCount);
        Debug.Assert(OuputLogits);

        foreach (var i in ..snapshot.Input.RowCount)
        {
            VectorHelper.MultiplyToMatrixAddTo(outputGradient.RowRef(i), snapshot.Input.RowRef(i), gradients.UnembeddingMatrix);
            UnembeddingMatrix.MultiplyTransposedTo(outputGradient.RowRef(i), snapshot.InputGradient.RowRef(i));
        }

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

                UnembeddedStorage.SetCount(field.RowCount, module.TokenCount);
                InputGradientStorage.OfSize(field);
            }
        }
        public Matrix Unembedded => UnembeddedStorage.Tensor;
        public Matrix InputGradient => InputGradientStorage.Tensor;

        private readonly Dynamic<Matrix> UnembeddedStorage = new();
        private readonly Dynamic<Matrix> InputGradientStorage = new();
    }

    [GeneratedAdam(typeof(IndexUnembeddingModule))]
    public sealed partial class Adam;
}
