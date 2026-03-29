using System.Numerics.Tensors;
using ML.Core.Attributes;
using ML.Core.Training;

namespace ML.Core.Modules;

[GeneratedModule(IncludeSerializer: true)]
public sealed partial class IndexEmbeddingModule(Matrix embeddingMatrix) : IInputModule<int[], Vector>
{
    [Weights] public Matrix EmbeddingMatrix { get; } = embeddingMatrix;

    public int TokenCount => EmbeddingMatrix.RowCount;
    public int EmbeddingSize => EmbeddingMatrix.ColumnCount;

    public IndexEmbeddingModule(int tokenCount, int embeddingSize)
        : this(Matrix.Create(tokenCount, embeddingSize)) { }

    public Vector Forward(int[] input, Snapshot snapshot)
    {
        snapshot.Input = input;
        var output = Matrix.Of(input.Length, EmbeddingSize, snapshot.Output);
        Debug.Assert(output.FlatCount == snapshot.Output.Count);

        foreach (var i in ..input.Length)
        {
            GetEmbedding(input[i]).CopyTo(output.RowSpan(i));
        }

        return snapshot.Output;
    }

    public Vector Backward(Vector outputGradients, Snapshot snapshot, Gradients gradients)
    {
        var outputGradientsMatrix = Matrix.Of(snapshot.Input.Length, EmbeddingSize, outputGradients);
        foreach (var i in ..snapshot.Input.Length)
        {
            var token = snapshot.Input[i];
            gradients.TouchedTokens.Add(token);
            var embeddingGradient = gradients.EmbeddingMatrix.RowSpan(token);
            TensorPrimitives.Add(embeddingGradient, outputGradientsMatrix.RowSpan(i), embeddingGradient);
        }

        return Vector.Empty;
    }

    private Span<Weight> GetEmbedding(int index)
    {
        if (index < 0 || index >= EmbeddingMatrix.RowCount)
        {
            throw new ArgumentException($"Unknown token: {index}");
        }

        return EmbeddingMatrix.RowSpan(index);
    }

    static IndexEmbeddingModule()
    {
        AdamOptimizer.Registry.Register<IndexEmbeddingModule>(static (op, module) => new Adam(op, module));
    }


    partial class Snapshot
    {
        public int[] Input
        {
            get;
            set
            {
                field = value;
                OutputStorage.SetCount(field.Length * module.EmbeddingSize);
            }
        } = [];
        public Vector Output => OutputStorage.Vector;


        private DynamicVector OutputStorage { get; }
    }

    partial class Gradients
    {
        // TODO: clear in reset (remove clear call from Adam.Apply)
        public HashSet<int> TouchedTokens { get; } = [];
    }

    public partial class Adam(AdamOptimizer optimizer, IndexEmbeddingModule module) : IModuleOptimizer<Gradients>
    {
        public IndexEmbeddingModule Module { get; } = module;
        public AdamOptimizer Optimizer { get; } = optimizer;

        public Matrix FirstMomentEmbeddingMatrix { get; } = Matrix.OfSize(module.EmbeddingMatrix);
        public Matrix SecondMomentEmbeddingMatrix { get; } = Matrix.OfSize(module.EmbeddingMatrix);

        public void Apply(Gradients gradients)
        {
            foreach (var token in gradients.TouchedTokens)
            {
                var gradient = gradients.EmbeddingMatrix.RowSpan(token);
                var firstMoment = FirstMomentEmbeddingMatrix.RowSpan(token);
                var secondMoment = SecondMomentEmbeddingMatrix.RowSpan(token);
                var weights = Module.EmbeddingMatrix.RowSpan(token);

                SpanOperations.MapTo(Optimizer.FirstMomentEstimateOperation, firstMoment, gradient, firstMoment);
                SpanOperations.MapTo(Optimizer.SecondMomentEstimateOperation, secondMoment, gradient, secondMoment);
                SpanOperations.MapTo(Optimizer.WeightReductionOperation, weights, firstMoment, secondMoment, weights);
            }

            NumericsDebug.AssertValidNumbers(FirstMomentEmbeddingMatrix);
            NumericsDebug.AssertValidNumbers(SecondMomentEmbeddingMatrix);

            gradients.TouchedTokens.Clear(); // TODO: this should happen in Gradients.FullReset();
        }

        public void FullReset()
        {
            FirstMomentEmbeddingMatrix.ResetZero();
            SecondMomentEmbeddingMatrix.ResetZero();
        }
    }
}
