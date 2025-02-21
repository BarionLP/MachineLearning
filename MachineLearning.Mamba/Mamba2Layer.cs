using System.Diagnostics;
using MachineLearning.Model.Initialization;
using MachineLearning.Model.Layer;
using MachineLearning.Model.Layer.Initialization;
using MachineLearning.Model.Layer.Snapshot;
using static MachineLearning.Serialization.ModelSerializationHelper;

namespace MachineLearning.Mamba;

public sealed class Mamba2Layer(int sequenceLength, int stateDimensions) : ILayer
{
    public int SequenceLength /*T*/ { get; } = sequenceLength;
    public int StateDimensions /*N*/ { get; } = stateDimensions;
    public Vector Alpha { get; } = Vector.Create(sequenceLength); // how much memory to keep from the previous step
    public Matrix B { get; } = Matrix.Create(sequenceLength, stateDimensions); // how does the input_t affect the memory h_t
    public Matrix C { get; } = Matrix.Create(sequenceLength, stateDimensions); // how does the memory h_t affect the output_t

    public Vector Forward(Vector input, Snapshot snapshot)
    {
        Debug.Assert(input.Count == SequenceLength);

        input.CopyTo(snapshot.Input);
        snapshot.Memory.ResetZero();

        for (int t = 0; t < SequenceLength; t++)
        {
            // h = alpha_t * h + B_t * x_t
            var h = snapshot.Memory.RowRef(t);
            if (t > 0)
            {
                snapshot.Memory.RowRef(t - 1).MultiplyTo(Alpha[t], h);
            }

            h.AddToSelf(B.RowRef(t).Multiply(input[t])); // add B_t * x_t

            // output[t] = C_t^T * h
            snapshot.Output[t] = C.RowRef(t).Dot(h);
        }

        return snapshot.Output;
    }

    public Vector BackwardPass(Snapshot snapshot, Vector outputGradient)
    {
        Debug.Assert(outputGradient.Count == SequenceLength);

        snapshot.GradientAlpha.ResetZero();
        snapshot.GradientInput.ResetZero();
        snapshot.GradientB.ResetZero();
        snapshot.GradientC.ResetZero();
        snapshot.GradientMemory.ResetZero();

        for (int t = SequenceLength - 1; t >= 0; t--)
        {
            // dY = derivative of L wrt outputY[t]
            float dY = outputGradient[t];

            // (a) output[t] = C[t] * H[t]
            // => dC[t] += H[t] * dY
            // => dH[t] += C[t] * dY
            snapshot.GradientC.RowRef(t).AddToSelf(snapshot.Memory.RowRef(t).Multiply(dY));
            snapshot.GradientMemory.RowRef(t).AddToSelf(C.RowRef(t).Multiply(dY));

            // (b) h[t] = alpha[t] * h[t-1] + B[t] * inputX[t]
            // => partial wrt alpha[t] = (h[t-1] dot dH[t])
            // => partial wrt h[t-1]  += alpha[t] * dH[t]
            // => partial wrt B[t]    += dH[t] * inputX[t]
            // => partial wrt inputX[t] = (B[t] dot dH[t])

            // derivative w.r.t alpha[t]
            // h[t-1] = (t>0) ? st.h[t-1] : zero
            var hPrev = (t == 0)
                           ? Vector.Create(StateDimensions)
                           : snapshot.Memory.RowRef(t - 1);

            snapshot.GradientAlpha[t] += hPrev.Dot(snapshot.GradientMemory.RowRef(t));  // dAlpha

            // derivative w.r.t H[t-1]
            // if t>0, add alpha[t]*dH[t] to dH[t-1]
            if (t > 0)
            {
                snapshot.GradientMemory.RowRef(t - 1).AddToSelf(snapshot.GradientMemory.RowRef(t).Multiply(Alpha[t]));
            }

            // derivative w.r.t. B[t] and input[t]
            // dB[t] = input[t] * dH[t]
            snapshot.GradientB.RowRef(t).AddToSelf(snapshot.GradientMemory.RowRef(t).Multiply(snapshot.Input[t]));

            snapshot.GradientInput[t] += B.RowRef(t).Dot(snapshot.GradientMemory.RowRef(t));  // partial w.r.t. input[t]

        }

        return snapshot.GradientInput;
    }

    public long ParameterCount { get; }
    public ILayerSnapshot CreateSnapshot() => new Snapshot(SequenceLength, StateDimensions);

    public sealed class Snapshot(int T, int N) : ILayerSnapshot
    {
        public Vector Input { get; } = Vector.Create(T);
        public Vector Output { get; } = Vector.Create(T);
        public Vector GradientInput { get; } = Vector.Create(T);

        public Matrix Memory /*H*/ { get; } = Matrix.Create(T, N); // one row per timestep t
        public Matrix GradientMemory { get; } = Matrix.Create(T, N);


        public Vector GradientAlpha { get; } = Vector.Create(T);
        public Matrix GradientB { get; } = Matrix.Create(T, N);
        public Matrix GradientC { get; } = Matrix.Create(T, N);

        public void Reset()
        {
            Input.ResetZero();
            Output.ResetZero();
            GradientInput.ResetZero();
            Memory.ResetZero();
            GradientMemory.ResetZero();
            GradientAlpha.ResetZero();
            GradientB.ResetZero();
            GradientC.ResetZero();
        }
    }

    public sealed class Initializer(Random? random = null) : IInitializer<Mamba2Layer>
    {
        public Random Random { get; } = random ?? Random.Shared;

        public void Initialize(Mamba2Layer layer)
        {
            float scale = 1.0f / MathF.Sqrt(layer.StateDimensions);

            // affects how much memory the model can keep from the previous step
            // optimally [0.9,1.0] must be [0,1] to prevent vanishing/exploding gradients
            layer.Alpha.Fill(0.9f);

            layer.B.MapToSelf(_ => InitializationHelper.RandomInNormalDistribution(Random, 0f, scale));
            layer.C.MapToSelf(_ => InitializationHelper.RandomInNormalDistribution(Random, 0f, scale));
        }
    }

}

public sealed class EmbeddedMamba2Layer(Vector alpha, Matrix b, Matrix c) : ILayer
{
    public int SequenceLength /*T*/ => Alpha.Count;
    public int StateDimensions /*N*/ => B.RowCount;
    public int EmbeddingDimensions /*E*/ => B.ColumnCount;

    // how much memory to keep from the previous step
    public Vector Alpha { get; } = alpha;

    // both could be a tensor of (T*N*E) but it makes sense to share this transformation across steps so only (N*E)
    public Matrix B { get; } = b; // how does the input_t affect the memory h_t
    public Matrix C { get; } = c; // how does the memory h_t affect the output_t

    public EmbeddedMamba2Layer(int sequenceLength, int stateDimensions, int embeddingDimensions)
        : this(Vector.Create(sequenceLength), Matrix.Create(stateDimensions, embeddingDimensions), Matrix.Create(stateDimensions, embeddingDimensions)) { }

    public Matrix Forward(Matrix input, Snapshot snapshot)
    {
        Debug.Assert(input.RowCount == SequenceLength);
        Debug.Assert(input.ColumnCount == EmbeddingDimensions);

        input.CopyTo(snapshot.Input);
        snapshot.Memory.ResetZero();

        for (int t = 0; t < SequenceLength; t++)
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

        return snapshot.Output;
    }

    private readonly Vector Zero = Vector.Create(b.RowCount);
    public Matrix BackwardPass(Snapshot snapshot, Matrix outputGradient)
    {
        Debug.Assert(outputGradient.RowCount == SequenceLength);
        Debug.Assert(outputGradient.ColumnCount == EmbeddingDimensions);

        snapshot.GradientAlpha.ResetZero();
        snapshot.GradientInput.ResetZero();
        snapshot.GradientB.ResetZero();
        snapshot.GradientC.ResetZero();
        snapshot.GradientMemory.ResetZero();

        for (int t = SequenceLength - 1; t >= 0; t--)
        {
            var outputGradient_t = outputGradient.RowRef(t);

            // output[t] = C^T * H[t]
            // => dC += H[t] * dY
            // => dH[t] += C * dY
            VectorHelper.MultiplyToMatrixAddTo(snapshot.Memory.RowRef(t), outputGradient_t, snapshot.GradientC);
            C.MultiplyAddTo(outputGradient_t, snapshot.GradientMemory.RowRef(t));

            // h[t] = alpha[t] * h[t-1] + B * input[t]
            // => wrt alpha[t] = (h[t-1] dot dH[t])
            // => wrt h[t-1]   += alpha[t] * dH[t]
            // => wrt B        += dH[t] * input[t]
            // => wrt input[t] = B^T * dH[t]

            // derivative wrt alpha[t]
            // h[t-1] = (t>0) ? st.h[t-1] : zero
            var hPrev = (t == 0) ? Zero : snapshot.Memory.RowRef(t - 1);

            snapshot.GradientAlpha[t] = hPrev.Dot(snapshot.GradientMemory.RowRef(t));

            // derivative wrt H[t-1]
            // if t>0, add alpha[t]*dH[t] to dH[t-1]
            if (t > 0)
            {
                snapshot.GradientMemory.RowRef(t).MultiplyTo(Alpha[t], snapshot.GradientMemory.RowRef(t - 1));
            }

            // derivative wrt B[t] and input[t]
            // dB[t] = input[t] * dH[t]
            VectorHelper.MultiplyToMatrixAddTo(snapshot.GradientMemory.RowRef(t), snapshot.Input.RowRef(t), snapshot.GradientB);

            B.MultiplyTransposedTo(snapshot.GradientMemory.RowRef(t), snapshot.GradientInput.RowRef(t));

        }

        snapshot.GradientC.DivideToSelf(SequenceLength);
        snapshot.GradientB.DivideToSelf(SequenceLength);

        return snapshot.GradientInput;
    }

    public static ErrorState Save(EmbeddedMamba2Layer layer, BinaryWriter writer)
    {
        writer.Write(layer.SequenceLength);
        writer.Write(layer.StateDimensions);
        writer.Write(layer.EmbeddingDimensions);
        WriteVector(layer.Alpha, writer);
        WriteMatrix(layer.B, writer);
        WriteMatrix(layer.C, writer);

        return default;
    }
    
    public static Result<EmbeddedMamba2Layer> Read(BinaryReader reader)
    {
        var sequenceLength = reader.ReadInt32();
        var stateDimensions = reader.ReadInt32();
        var embeddingDimensions = reader.ReadInt32();
        var alpha = ReadVector(sequenceLength, reader);
        var b = ReadMatrix(stateDimensions, embeddingDimensions, reader);
        var c = ReadMatrix(stateDimensions, embeddingDimensions, reader);
        return new EmbeddedMamba2Layer(alpha, b, c);
    }

    public long ParameterCount => Alpha.Count + B.FlatCount + C.FlatCount;
    public ILayerSnapshot CreateSnapshot() => new Snapshot(SequenceLength, StateDimensions, EmbeddingDimensions);

    public sealed class Snapshot(int T, int N, int E) : ILayerSnapshot
    {
        public Matrix Input { get; } = Matrix.Create(T, E);
        public Matrix Output { get; } = Matrix.Create(T, E);
        public Matrix GradientInput { get; } = Matrix.Create(T, E);

        public Matrix Memory /*H*/ { get; } = Matrix.Create(T, N); // one row per timestep
        public Matrix GradientMemory { get; } = Matrix.Create(T, N);

        public Vector GradientAlpha { get; } = Vector.Create(T);
        public Matrix GradientB { get; } = Matrix.Create(N, E);
        public Matrix GradientC { get; } = Matrix.Create(N, E);

        public void Reset()
        {
            Input.ResetZero();
            Output.ResetZero();
            GradientInput.ResetZero();
            Memory.ResetZero();
            GradientMemory.ResetZero();
            GradientAlpha.ResetZero();
            GradientB.ResetZero();
            GradientC.ResetZero();
        }
    }

    public sealed class Initializer(Random? random = null) : IInitializer<EmbeddedMamba2Layer>
    {
        public Random Random { get; } = random ?? Random.Shared;

        public void Initialize(EmbeddedMamba2Layer layer)
        {
            var scale = 1 / Weight.Sqrt(layer.StateDimensions);

            // affects how much memory the layer can keep from the previous step
            // optimally [0.9,1.0] must be [0,1] to prevent vanishing/exploding gradients
            layer.Alpha.Fill(0.9f);

            layer.B.MapToSelf(_ => InitializationHelper.RandomInNormalDistribution(Random, 0f, scale));
            layer.C.MapToSelf(_ => InitializationHelper.RandomInNormalDistribution(Random, 0f, scale));
        }
    }

}
