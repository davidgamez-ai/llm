/**
 * Computes simple (unscaled, unweighted) self-attention scores for a
 * sequence of embedding vectors. For each token, its embedding vector is
 * dotted with every other token's embedding vector (including its own),
 * producing a square matrix of raw attention scores between every pair of
 * tokens in the sequence. Each token's row of scores is then normalized
 * with softmax, turning the raw dot products into attention weights that
 * sum to 1.
 */
class SimpleAttention {
    /**
     * Takes the dot product of every embedding vector with every other
     * embedding vector, including itself, then normalizes each token's row of
     * scores with softmax. Row i, column j of the result is the share of
     * token i's attention that falls on token j.
     *
     * @param embeddings - Token embeddings as produced by Embedding.getEmbedding(): one row per token, each row embeddingSize values wide.
     * @returns A tokenCount x tokenCount matrix of attention weights, where entry [i][j] is token i's normalized attention score for token j, and each row sums to 1.
     */
    calculate(embeddings) {
        // Step 1: note the shape of the input so the calculation can be
        // followed from the start.
        console.log(`SimpleAttention| Calculating attention scores for ${embeddings.length} tokens, each embedding ${embeddings[0]?.length ?? 0} values wide.`);
        const rawScores = [];
        // Step 2: for each token in turn, compare its embedding vector against
        // every token's embedding vector (including its own).
        for (let i = 0; i < embeddings.length; i++) {
            const vectorA = embeddings[i];
            const scoresForToken = [];
            for (let j = 0; j < embeddings.length; j++) {
                const vectorB = embeddings[j];
                // Step 3: reduce the pair of vectors, element by element, into a
                // single dot-product score, which is that pair's raw attention score.
                const dotProduct = vectorA.reduce((sum, value, index) => sum + value * vectorB[index], 0);
                scoresForToken.push(dotProduct);
            }
            rawScores.push(scoresForToken);
            console.log(`SimpleAttention| Raw scores for token ${i}:`, scoresForToken);
        }
        // Step 4: normalize each token's row of raw scores with softmax, so it
        // sums to 1. Each score is exponentiated, after subtracting the row's
        // max score from every value first; this shifts the largest exponent
        // down to exp(0) = 1, which avoids overflow for large scores without
        // changing the resulting ratios, since it's equivalent to dividing both
        // the numerator and denominator by the same constant, exp(max).
        const attentionWeights = rawScores.map((scoresForToken, i) => {
            const maxScore = Math.max(...scoresForToken);
            const exponentiatedScores = scoresForToken.map((score) => Math.exp(score - maxScore));
            const total = exponentiatedScores.reduce((sum, score) => sum + score, 0);
            const softmaxScores = exponentiatedScores.map((score) => score / total);
            console.log(`SimpleAttention| Softmax weights for token ${i}:`, softmaxScores);
            return softmaxScores;
        });
        console.log("SimpleAttention| Finished calculating attention scores.");
        return attentionWeights;
    }
}
export default SimpleAttention;
