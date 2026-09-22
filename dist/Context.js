/**
 * Builds context vectors from token embeddings and their attention scores.
 * Each token's context vector is a weighted blend of every token's
 * embedding vector, weighted by how much attention that token pays to
 * each other token, so it captures information from the whole sequence
 * rather than just the token itself.
 */
class Context {
    /**
     * For each token, scales every embedding vector in the sequence by that
     * token's attention weight for it, then sums the scaled vectors together
     * to produce that token's context vector.
     *
     * @param embeddings - Token embeddings as produced by Embedding.getEmbedding(): one row per token, each row embeddingSize values wide.
     * @param attentionScores - Attention weights as produced by SimpleAttention.calculate(): a tokenCount x tokenCount matrix where entry [i][j] is token i's attention weight for token j.
     * @returns One context vector per token, in token order, each embeddingSize values wide.
     */
    getContext(embeddings, attentionScores) {
        // Step 1: note the shape of the input so the calculation can be
        // followed from the start.
        console.log(`Context| Calculating context vectors for ${embeddings.length} tokens, each embedding ${embeddings[0]?.length ?? 0} values wide.`);
        const contextVectors = [];
        // Step 2: for each token in turn, build its context vector out of every
        // token's embedding vector, weighted by this token's attention scores.
        for (let i = 0; i < embeddings.length; i++) {
            const weightsForToken = attentionScores[i];
            // The context vector starts at zero and accumulates each weighted
            // embedding vector as the tokens are worked through.
            let contextVector = new Array(embeddings[0]?.length ?? 0).fill(0);
            for (let j = 0; j < embeddings.length; j++) {
                const embedding = embeddings[j];
                const weight = weightsForToken[j];
                // Step 3: scale this token's embedding vector by its attention
                // weight, then add the scaled vector into the running total.
                const weightedEmbedding = embedding.map((value) => value * weight);
                contextVector = contextVector.map((value, index) => value + weightedEmbedding[index]);
            }
            contextVectors.push(contextVector);
            console.log(`Context| Context vector for token ${i}:`, contextVector);
        }
        console.log("Context| Finished calculating context vectors.");
        return contextVectors;
    }
}
export default Context;
