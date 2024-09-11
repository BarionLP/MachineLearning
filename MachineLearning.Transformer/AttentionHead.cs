using Ametrin.Numerics;
using MachineLearning.Model.Initialization;
using System.Diagnostics;

namespace MachineLearning.Transformer;

public sealed class AttentionHead(ModelInfo Info)
{
    public ModelInfo Info { get; } = Info;
    public Matrix KeyWeights /*W_K*/ { get; } = Matrix.Create(Info.KeyQueryDimensions, Info.EmbeddingDimensions);
    public Matrix QueryWeights /*W_Q*/ { get; } = Matrix.Create(Info.KeyQueryDimensions, Info.EmbeddingDimensions);
    public Matrix ValueDownWeights /*W_V*/ { get; } = Matrix.Create(Info.KeyQueryDimensions, Info.EmbeddingDimensions);
    public Matrix ValueUpWeights { get; } = Matrix.Create(Info.EmbeddingDimensions, Info.KeyQueryDimensions);

    private readonly Weight QueryDimensionsSqrt = Math.Sqrt(Info.KeyQueryDimensions);

    public Matrix GetEmbeddingDelta(HeadPass pass)
    {   
        pass.Deconstruct(out var input, out var keyEmbedding, out var queryEmbedding, out var valueVectors, out var attentionPattern);
        
        Debug.Assert(input.RowCount == Info.ContextSize);
        Debug.Assert(input.ColumnCount == Info.EmbeddingDimensions);

        // calculate key and query matrices
        input.MultiplyRowwiseTo(KeyWeights, keyEmbedding);
        input.MultiplyRowwiseTo(QueryWeights, queryEmbedding);
        
        var valueDown = Vector.Create(Info.KeyQueryDimensions);
        for(int i = 0; i < input.RowCount; i++)
        {
            //calculate the change to a token vector v this token vector would produce if this token attends to v
            //to be more efficient the change gets projected into the query space and back to the embedding space
            ValueDownWeights.MultiplyTo(input.RowRef(i), valueDown);
            ValueUpWeights.MultiplyTo(valueDown, valueVectors.RowRef(i));
            // the following can be done with the down projection and projected back up to save on computation (inside the attention block).
        }

        // the dot product of each query with each key defines how much the token of the query should be affected by the token of the key
        // high dot products mean the key token "attends to" the query token
        // rows are the queries and columns the keys
        for(int queryIndex = 0; queryIndex < input.RowCount; queryIndex++)
        {
            var vector = attentionPattern.RowRef(queryIndex);
            for(int keyIndex = 0; keyIndex < input.RowCount; keyIndex++)
            {
                //force attendance to 0 (-infty before activation) for earlier tokes, allows training on the whole sentence instead just the next word (can be disabled after training)
                vector[keyIndex] = keyIndex > queryIndex ? double.NegativeInfinity : keyEmbedding.RowRef(keyIndex).Dot(queryEmbedding.RowRef(queryIndex)) / QueryDimensionsSqrt; // divide by sqrt(QueryDimensions) for numerical stability
            }
            // softmax each row to get a percentage distribution on how much each key token should affect each query token
            vector.SoftMaxInPlace();
        }

        //Console.WriteLine(attentionPattern);

        var deltaMatrix = Matrix.Create(input.RowCount, Info.EmbeddingDimensions);
        for (int tokenIndex = 0; tokenIndex < input.RowCount; tokenIndex++)
        {
            var deltaVector = deltaMatrix.RowRef(tokenIndex);
            for (int deltaIndex = 0; deltaIndex < input.RowCount; deltaIndex++)
            {
                var valueVector = valueVectors.RowRef(deltaIndex).Multiply(attentionPattern[tokenIndex, deltaIndex]); // cannot operate on reference because it will be used later again (?)
                deltaVector.AddToSelf(valueVector);
            }
        }

        return deltaMatrix;
    }


    public sealed record BackpropagationContext(Matrix keyEmbedding, Matrix queryEmbedding, Matrix attentionPattern, Matrix valueVectors);
    public void Backpropagate(Matrix input, Matrix dNextDeltaMatrix, BackpropagationContext context)
    {
        var dKeyWeights = Matrix.OfSize(KeyWeights);
        var dQueryWeights = Matrix.OfSize(QueryWeights);
        var dValueDownWeights = Matrix.OfSize(ValueDownWeights);
        var dValueUpWeights = Matrix.OfSize(ValueUpWeights);
        var dInput = Matrix.OfSize(input);


        var dValueVectors = Matrix.OfSize(input);
        var dAttentionPattern = Matrix.OfSize(context.attentionPattern);

        var tmp = Vector.Create(input.ColumnCount);
        // Backpropagate through the output aggregation step
        for(int tokenIndex = 0; tokenIndex < input.RowCount; tokenIndex++)
        {
            var dDeltaVector = dNextDeltaMatrix.RowRef(tokenIndex);
            for(int deltaIndex = 0; deltaIndex < input.RowCount; deltaIndex++)
            {
                dAttentionPattern[tokenIndex, deltaIndex] += dDeltaVector.Dot(context.valueVectors.RowRef(deltaIndex));
                dDeltaVector.MultiplyTo(context.attentionPattern[tokenIndex, deltaIndex], tmp);
                dValueVectors.RowRef(deltaIndex).AddToSelf(tmp);
            }
        }

        //attention pattern calculation:
        var dKeyEmbedding = Matrix.Create(context.keyEmbedding.RowCount, context.keyEmbedding.ColumnCount);
        var dQueryEmbedding = Matrix.Create(context.queryEmbedding.RowCount, context.queryEmbedding.ColumnCount);
        for(int queryIndex = 0; queryIndex < input.RowCount; queryIndex++)
        {
            var dAttentionRow = dAttentionPattern.RowRef(queryIndex);
            //Apply Softmax gradient backpropagation to dAttentionRow in place
            for(int keyIndex = 0; keyIndex < input.RowCount; keyIndex++)
            {
                if(keyIndex <= queryIndex)
                {
                    context.queryEmbedding.RowRef(queryIndex).MultiplyTo(dAttentionRow[keyIndex] / QueryDimensionsSqrt, tmp);
                    dKeyEmbedding.RowRef(keyIndex).AddToSelf(tmp);
                    
                    context.keyEmbedding.RowRef(keyIndex).MultiplyTo(dAttentionRow[keyIndex] / QueryDimensionsSqrt, tmp);
                    dQueryEmbedding.RowRef(queryIndex).AddToSelf(tmp);
                }
            }
        }

        var dInputKey = Matrix.OfSize(input);
        var dInputQuery = Matrix.OfSize(input);

        //var tmp_m = Matrix.Create(dQueryEmbedding.ColumnCount, )
        for(int i = 0; i < input.RowCount; i++)
        {
            QueryWeights.MultiplyTo(dQueryEmbedding.RowRef(i), tmp);
            dInputQuery.RowRef(i).AddToSelf(tmp);
            dQueryWeights.AddToSelf(VectorHelper.MultiplyToMatrix(dQueryEmbedding.RowRef(i), input.RowRef(i)));


            //ValueDownWeights.Multiply(input.RowRef(i), valueDown);
            //ValueUpWeights.Multiply(valueDown, valueVectors.RowRef(i));
        }
        //- For each i in input.RowCount:
        //       -Backpropagate through QueryWeights: dInputQuery.row(i) += QueryWeights.T * dQueryEmbedding.row(i)
        //       - dQueryWeights += dQueryEmbedding.row(i) * input.row(i).T
        //       - Backpropagate through KeyWeights: dInputKey.row(i) += KeyWeights.T * dKeyEmbedding.row(i)
        //       - dKeyWeights += dKeyEmbedding.row(i) * input.row(i).T
        //   - dInput += dInputKey + dInputQuery
    }

    public sealed record HeadPass(Matrix input, Matrix keyEmbedding, Matrix queryEmbedding, Matrix valueVectors, Matrix attentionPattern)
    {
        public static HeadPass Allocate(ModelInfo info)
        {
            return new HeadPass(
                    input: Matrix.Create(info.ContextSize, info.EmbeddingDimensions),
                    keyEmbedding: Matrix.Create(info.ContextSize, info.KeyQueryDimensions),
                    queryEmbedding: Matrix.Create(info.ContextSize, info.KeyQueryDimensions),
                    valueVectors: Matrix.Create(info.ContextSize, info.EmbeddingDimensions),
                    attentionPattern: Matrix.CreateSquare(info.ContextSize)
                );
        }
    }

    public Matrix BackwardPass(Matrix dDeltaMatrix, HeadPass context)
    {
        var (input, keyEmbedding, queryEmbedding, valueVectors, attentionPattern) = context;
        
        var dKeyWeights = Matrix.OfSize(KeyWeights);
        var dQueryWeights = Matrix.OfSize(QueryWeights);
        var dValueDownWeights = Matrix.OfSize(ValueDownWeights);
        var dValueUpWeights = Matrix.OfSize(ValueUpWeights);
        var dInput = Matrix.OfSize(input);

        var dValueVectors = Matrix.OfSize(valueVectors);
        var dAttentionPattern = Matrix.OfSize(attentionPattern);

        // Step 3: Backpropagate through the output aggregation step
        for(int tokenIndex = 0; tokenIndex < input.RowCount; tokenIndex++)
        {
            var dDeltaVector = dDeltaMatrix.RowRef(tokenIndex);
            for(int deltaIndex = 0; deltaIndex < input.RowCount; deltaIndex++)
            {
                dAttentionPattern[tokenIndex, deltaIndex] += dDeltaVector.Dot(valueVectors.RowRef(deltaIndex));
                dValueVectors.RowRef(deltaIndex).AddToSelf(dDeltaVector.Multiply(attentionPattern[tokenIndex, deltaIndex]));
            }
        }

        // Step 4: Backpropagate through the attention pattern calculation
        var dKeyEmbedding = Matrix.OfSize(keyEmbedding);
        var dQueryEmbedding = Matrix.OfSize(queryEmbedding);

        for(int queryIndex = 0; queryIndex < input.RowCount; queryIndex++)
        {
            var dAttentionRow = dAttentionPattern.RowRef(queryIndex);
            var attentionRow = attentionPattern.RowRef(queryIndex);

            // Apply Softmax gradient backpropagation
            MatrixHelper.SoftMaxGradientInPlace(dAttentionRow, attentionRow);

            for(int keyIndex = 0; keyIndex < input.RowCount; keyIndex++)
            {
                if(keyIndex <= queryIndex)
                {
                    dKeyEmbedding.RowRef(keyIndex).AddToSelf(queryEmbedding.RowRef(queryIndex).Multiply(dAttentionRow[keyIndex] / QueryDimensionsSqrt));
                    dQueryEmbedding.RowRef(queryIndex).AddToSelf(keyEmbedding.RowRef(keyIndex).Multiply(dAttentionRow[keyIndex] / QueryDimensionsSqrt));
                }
            }
        }

        // Step 5: Backpropagate through key, query, and value weight multiplications
        var dInputKey = Matrix.OfSize(input);
        var dInputQuery = Matrix.OfSize(input);

        for(int i = 0; i < input.RowCount; i++)
        {
            var dQueryEmbedRow = dQueryEmbedding.RowRef(i);
            var dKeyEmbedRow = dKeyEmbedding.RowRef(i);
            var inputRow = input.RowRef(i);

            // Backpropagate through QueryWeights
            MatrixHelper.MultiplyTransposeWithGradient(dQueryEmbedRow, inputRow, dQueryWeights);
            dInputQuery.RowRef(i).AddToSelf(QueryWeights.MultiplyTransposed(dQueryEmbedRow));

            // Backpropagate through KeyWeights
            MatrixHelper.MultiplyTransposeWithGradient(dKeyEmbedRow, inputRow, dKeyWeights);
            dInputKey.RowRef(i).AddToSelf(KeyWeights.MultiplyTransposed(dKeyEmbedRow));
        }

        dInput.AddToSelf(dInputKey);
        dInput.AddToSelf(dInputQuery);

        // Step 6: Backpropagate through value weight multiplications
        var dValueDown = Vector.Create(Info.KeyQueryDimensions);
        for(int i = 0; i < input.RowCount; i++)
        {
            var dValueVectorRow = dValueVectors.RowRef(i);
            var inputRow = input.RowRef(i);

            // Backpropagate through ValueUpWeights
            ValueUpWeights.MultiplyTransposedTo(dValueVectorRow, dValueDown);
            MatrixHelper.MultiplyTransposeWithGradient(dValueVectorRow, dValueDown, dValueUpWeights);

            // Backpropagate through ValueDownWeights
            ValueDownWeights.MultiplyTransposedTo(dValueDown, dInput.RowRef(i));
            MatrixHelper.MultiplyTransposeWithGradient(dValueDown, inputRow, dValueDownWeights);
        }

        return dInput;
    }


    public void Initialize(IInitializer initializer)
    {
        initializer.Initialize(KeyWeights);
        initializer.Initialize(QueryWeights);
        initializer.Initialize(ValueDownWeights);
        initializer.Initialize(ValueUpWeights);
    }

    public int GetWeightsCount() {
        return 
            Info.KeyQueryDimensions * Info.EmbeddingDimensions * 2 + // key & query weights
            Info.KeyQueryDimensions * Info.EmbeddingDimensions + //value down
            Info.EmbeddingDimensions * Info.KeyQueryDimensions //value up
        ;
    }
}