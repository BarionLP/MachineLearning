using MachineLearning.Model.Attributes;
using MachineLearning.Model.Initialization;
using MachineLearning.Model.Layer;
using MachineLearning.Model.Layer.Initialization;
using MachineLearning.Serialization;
using MachineLearning.Training.Attributes;

namespace MachineLearning.Mamba;

[GeneratedLayer, LayerSerializer("vmam2", 2), GenerateOptimizers]
public sealed partial class Mamba2VectorLayer : ILayer<Matrix, Mamba2VectorLayer.Snapshot>
{
    public int MaxSequenceLength /*T*/ => Alpha.Count;
    public int StateDimensions /*N*/ => B.RowCount;
    public int EmbeddingDimensions /*E*/ => B.ColumnCount;

    // how much memory to keep from the previous step
    [Weights] public Vector Alpha { get; }

    // both could be a tensor (T*N*E) but it makes sense to share this transformation across steps so only (N*E)
    [Weights] public Matrix B { get; } // how does the input_t affect the memory h_t
    [Weights] public Matrix C { get; } // how does the memory h_t affect the output_t

    public Mamba2VectorLayer(int sequenceLength, int stateDimensions, int embeddingDimensions)
        : this(Vector.Create(sequenceLength), Matrix.Create(stateDimensions, embeddingDimensions), Matrix.Create(stateDimensions, embeddingDimensions)) { }

    public Matrix Forward(Matrix input, Snapshot snapshot)
    {
        Debug.Assert(input.RowCount <= MaxSequenceLength);
        Debug.Assert(input.ColumnCount == EmbeddingDimensions);

        snapshot.SequenceLength = input.RowCount;

        snapshot.Input = input;
        snapshot.Output = Matrix.OfSize(input);
        snapshot.Memory.ResetZero();

        for (int t = 0; t < input.RowCount; t++)
        {
            // h = alpha_t * h + B * x_t
            var h = snapshot.Memory.RowRef(t);
            if (t > 0)
            {
                snapshot.Memory.RowRef(t - 1).MultiplyTo(Alpha[t], h);
            }

            B.MultiplyAddTo(input.RowRef(t), h); // h += B * x_t

            // y_t = C^T * h
            C.MultiplyTransposedTo(h, snapshot.Output.RowRef(t));
        }

        NumericsDebug.AssertValidNumbers(snapshot.Output);

        return snapshot.Output;
    }

    private Vector Zero
    {
        get
        {
            field ??= Vector.Create(StateDimensions);
            return field;
        }
    }

    public Matrix Backward(Matrix outputGradient, Snapshot snapshot, Gradients gradients)
    {
        Debug.Assert(outputGradient.RowCount <= MaxSequenceLength);
        Debug.Assert(outputGradient.ColumnCount == EmbeddingDimensions);

        // snapshot.GradientInput.ResetZero();
        snapshot.GradientInput = Matrix.OfSize(outputGradient);
        snapshot.GradientMemory.ResetZero();

        for (int t = outputGradient.RowCount - 1; t >= 0; t--)
        {
            var outputGradient_t = outputGradient.RowRef(t);

            // output[t] = C^T * H[t]
            // => dC += H[t] * dY
            // => dH[t] += C * dY
            VectorHelper.MultiplyToMatrixAddTo(snapshot.Memory.RowRef(t), outputGradient_t, gradients.C);
            C.MultiplyAddTo(outputGradient_t, snapshot.GradientMemory.RowRef(t));

            // h[t] = alpha[t] * h[t-1] + B * input[t]
            // => wrt alpha[t] = (h[t-1] dot dH[t])
            // => wrt h[t-1]   += alpha[t] * dH[t]
            // => wrt B        += dH[t] * input[t]
            // => wrt input[t] = B^T * dH[t]

            // derivative wrt alpha[t]
            // h[t-1] = (t>0) ? st.h[t-1] : zero
            var hPrev = (t == 0) ? Zero : snapshot.Memory.RowRef(t - 1);

            gradients.Alpha[t] = hPrev.Dot(snapshot.GradientMemory.RowRef(t));

            // derivative wrt H[t-1]
            // if t>0, add alpha[t]*dH[t] to dH[t-1]
            if (t > 0)
            {
                snapshot.GradientMemory.RowRef(t).MultiplyTo(Alpha[t], snapshot.GradientMemory.RowRef(t - 1));
            }

            // derivative wrt B[t] and input[t]
            // dB[t] = input[t] * dH[t]
            VectorHelper.MultiplyToMatrixAddTo(snapshot.GradientMemory.RowRef(t), snapshot.Input.RowRef(t), gradients.B);

            B.MultiplyTransposedTo(snapshot.GradientMemory.RowRef(t), snapshot.GradientInput.RowRef(t));

        }

        // snapshot.GradientC.DivideToSelf(SequenceLength);
        // snapshot.GradientB.DivideToSelf(SequenceLength);

        return snapshot.GradientInput;
    }

    public partial class Snapshot
    {
        public int SequenceLength { get; set; }
        // public Matrix Input { get; } = Matrix.Create(layer.MaxSequenceLength, layer.EmbeddingDimensions);
        public Matrix Input { get; set; }
        // public Matrix GradientInput { get; } = Matrix.Create(layer.MaxSequenceLength, layer.EmbeddingDimensions);
        public Matrix GradientInput { get; set; }
        // public Matrix Output { get; } = Matrix.Create(layer.MaxSequenceLength, layer.EmbeddingDimensions);
        public Matrix Output { get; set; }

        public Matrix Memory /*H*/ { get; } = Matrix.Create(layer.MaxSequenceLength, layer.StateDimensions); // one row per timestep
        public Matrix GradientMemory { get; } = Matrix.Create(layer.MaxSequenceLength, layer.StateDimensions);
    }

    public sealed class Initializer(Random? random = null) : IInitializer<Mamba2VectorLayer>
    {
        public Random Random { get; } = random ?? Random.Shared;

        public void Initialize(Mamba2VectorLayer layer)
        {
            var scale = Weight.Sqrt(6 / ((Weight)layer.StateDimensions + layer.EmbeddingDimensions));

            // affects how much memory the layer can keep from the previous step
            // optimally [0.9,1.0] must be [0,1] to prevent vanishing/exploding gradients
            layer.Alpha.Fill(0.9f);

            layer.B.MapToSelf(_ => InitializationHelper.RandomInUniformDistribution(Random, 0f, scale));
            layer.C.MapToSelf(_ => InitializationHelper.RandomInUniformDistribution(Random, 0f, scale));
            // foreach (var i in ..layer.B.RowCount)
            // {
            //     var row = layer.B.RowRef(i);
            //     var mag = row.Magnitude();
            //     Console.Write(mag);
            //     Console.Write(" -> ");
            //     row.DivideToSelf(mag / 0.3f);
            //     Console.WriteLine(row.Magnitude());
            // }
            // foreach (var i in ..layer.C.RowCount)
            // {
            //     var row = layer.C.RowRef(i);
            //     var mag = row.Magnitude();
            //     Console.Write(mag);
            //     Console.Write(" -> ");
            //     row.DivideToSelf(mag / 0.3f);
            //     Console.WriteLine(row.Magnitude());
            // }
        }
    }

}
