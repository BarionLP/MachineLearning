using System.Numerics.Tensors;
using ML.Core.Attributes;
using ML.Core.Training;

namespace ML.Core.Modules;

[GeneratedModule(IncludeSerializer: true)]
public sealed partial class IndexEmbeddingModule(Matrix embeddingMatrix) : IInputModule<int[], Matrix>
{
    [Weights] public Matrix EmbeddingMatrix { get; } = embeddingMatrix;

    public int TokenCount => EmbeddingMatrix.RowCount;
    public int EmbeddingSize => EmbeddingMatrix.ColumnCount;

    public IndexEmbeddingModule(int tokenCount, int embeddingSize)
        : this(Matrix.Create(tokenCount, embeddingSize)) { }

    public Matrix Forward(int[] input, Snapshot snapshot)
    {
        snapshot.Input = input;

        foreach (var i in ..input.Length)
        {
            GetEmbedding(input[i]).CopyTo(snapshot.Output.RowSpan(i));
        }

        return snapshot.Output;
    }

    public Matrix Backward(Matrix outputGradients, Snapshot snapshot, Gradients gradients)
    {
        foreach (var i in ..snapshot.Input.Length)
        {
            var token = snapshot.Input[i];
            gradients.TouchedTokens.Add(token);
            var embeddingGradient = gradients.EmbeddingMatrix.RowSpan(token);
            TensorPrimitives.Add(embeddingGradient, outputGradients.RowSpan(i), embeddingGradient);
        }

        return Matrix.Empty;
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
                Output = Matrix.Of(field.Length, module.EmbeddingSize, OutputStorage.Vector);
            }
        } = [];

        public Matrix Output { get; private set; }

        private readonly DynamicVector OutputStorage = new();

        private void OnDispose()
        {
            Output = Matrix.Empty;
        }
    }

    partial class Gradients
    {
        public HashSet<int> TouchedTokens { get; } = [];

        private void OnReset()
        {
            TouchedTokens.Clear();
        }
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
        }

        public void FullReset()
        {
            FirstMomentEmbeddingMatrix.ResetZero();
            SecondMomentEmbeddingMatrix.ResetZero();
        }
    }
}
