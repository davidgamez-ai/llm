/**
 * Cuts tokens into non-overlapping windows of windowLength tokens, as the
 * book's GPTDatasetV1 does with a stride equal to the context length. The
 * targets of each window are its inputs shifted one place to the right, so a
 * window needs windowLength + 1 tokens. Only full windows are made: any tokens
 * left over at the end are not used.
 *
 * @param tokens - The token ids to cut into windows.
 * @param windowLength - The number of inputs in each window.
 * @returns The windows, in the order they appear in tokens.
 */
export function createWindows(tokens, windowLength) {
    const windows = [];
    // A window starting at start needs tokens up to start + windowLength for its last target.
    for (let start = 0; start + windowLength < tokens.length; start += windowLength) {
        windows.push({
            inputs: tokens.slice(start, start + windowLength),
            targets: tokens.slice(start + 1, start + windowLength + 1),
        });
    }
    return windows;
}
/**
 * Joins documents (files or stories), already encoded as token ids, into one
 * list, with the <|endoftext|> token between each document and the next. This
 * is how GPT-2 marks the boundaries between documents: the model learns that
 * the text after the token is unrelated to the text before it. A single
 * document is returned unchanged, with no token added.
 *
 * @param documents - The token ids of each document, in order.
 * @param endOfTextId - The id of the <|endoftext|> token.
 * @returns The joined token ids.
 */
export function joinDocuments(documents, endOfTextId) {
    const tokens = [];
    for (let i = 0; i < documents.length; i++) {
        if (i > 0) {
            tokens.push(endOfTextId);
        }
        /*
         * Appended with a loop rather than push(...document), because spreading
         * a whole book's tokens as arguments can exceed the engine's argument
         * limit.
         */
        for (const token of documents[i]) {
            tokens.push(token);
        }
    }
    return tokens;
}
