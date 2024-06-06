using System.Diagnostics;
using System.Text;

namespace MachineLearning.Samples.Language;

public sealed class StringEmbedder(int contextSize) : IEmbedder<string, char>
{
    public Vector Embed(string input)
    {
        var bytes = Encoding.Latin1.GetBytes(input);
        var result = Vector.Create(8 * bytes.Length);

        if(input.Length != bytes.Length)
        {
            throw new UnreachableException();
        }

        for(var index = 0; index < bytes.Length; index++)
        {
            var b = bytes[index];
            for(var i = 0; i < 8; i++)
            {
                result[index * 8 + i] = (b & 1 << i) != 0 ? 1.0 : 0.0;
            }
        }

        return PadLeft(result, contextSize * 8);
    }
    public static Vector PadLeft(Vector vector, int totalWidth)
    {
        if(vector.Count >= totalWidth)
            return vector;

        var paddedVector = Vector.Create(totalWidth);
        vector.AsSpan().CopyTo(paddedVector[totalWidth - vector.Count, vector.Count]);
        return paddedVector;
    }

    public char UnEmbed(Vector input)
    {
        return LanguageDataSource.TOKENS[GetWeightedRandomIndex(input)];

        static int GetWeightedRandomIndex(Vector weights)
        {
           var value = Random.Shared.NextDouble();
           for(int i = 0; i < weights.Count; i++)
           {
               value -= weights[i];
               if(value < 0)
                   return i;
           }
           return weights.Count - 1;
        }

        static int IndexOfMax(Vector weights)
        {
            int max = 0;
            for(int i = 0; i < weights.Count; i++)
            {
                if(weights[i] > weights[max])
                {
                    max = i;
                }
            }

            return max;
        }
    }
}