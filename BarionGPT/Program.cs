using BarionGPT;

var info = new ModelInfo {
    EmbeddingDimensions = 128,
    ContextSize = 64,
    QueryDimensions = 32,
    AttentionHeadCount = 4,
    Temperature = 2,
};

var model = new Model(info);

var prompt = "hello";
char predicted;

Console.Write(prompt);

do {
    predicted = model.Process(prompt);
    prompt += predicted;
    Console.Write(predicted);
} while(predicted != '.' && prompt.Length <= info.ContextSize);

Console.WriteLine();

prompt = "my name is ";

Console.Write(prompt);

do {
    predicted = model.Process(prompt);
    prompt += predicted;
    Console.Write(predicted);
} while(predicted != '.' && prompt.Length <= info.ContextSize);

Console.WriteLine();

//var head = new AttentionHead(new ModelInfo() { 
//    EmbeddingDimensions = 4,
//    ContextSize = 4,
//    QueryDimensions = 2,
//    AttentionHeadCount = 1,
//});

//Console.WriteLine(head.GetEmbeddingDelta(DenseMatrix.CreateRandom(4, 3, info.InitialDistribution)));

//var input = DenseMatrix.CreateRandom(3, 3, new ContinuousUniform());
//var keys = DenseMatrix.CreateRandom(3, 3, new ContinuousUniform());

//Console.WriteLine(input*keys);


/*
Matrix multiplication m*n means multiplying every column of n by m; m*n != m*n
Matrix Vector multiplication m*v means taking the dot product of every row of m with v
 */

