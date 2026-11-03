import { test } from "node:test";
import assert from "node:assert/strict";
import BPETokenizer from "../BPETokenizer.js";
import Generator from "../Generator.js";
import Hyperparameters from "../Hyperparameters.js";
const tokenizer = new BPETokenizer();
/** The id of a token that is not the end token, which decodes to " the". */
const THE = tokenizer.encode(" the")[0];
/**
 * Creates a Generator whose model always chooses the given token ids in order,
 * using greedy sampling, then repeats the last. The model is a stand-in with
 * only the method Generator uses, so that which token comes next is certain.
 *
 * @param tokenIds - The ids the model chooses, in order.
 * @param outputLength - The number of tokens generate() produces.
 * @param contextLength - The model's context length.
 * @returns The generator, and the contexts the model was given, one per token generated.
 */
const generatorChoosing = (tokenIds, outputLength = 5, contextLength = 100) => {
    const hyperparameters = new Hyperparameters(8, contextLength, 0.1, Hyperparameters.GENERATION_MODE, 2, 1, 2);
    hyperparameters.outputLength = outputLength;
    hyperparameters.samplingMethod = Hyperparameters.GREEDY_SAMPLING;
    const contexts = [];
    const model = {
        update(context) {
            const chosen = tokenIds[Math.min(contexts.length, tokenIds.length - 1)];
            contexts.push(context);
            const logits = new Array(hyperparameters.vocabularySize).fill(0);
            logits[chosen] = 1;
            return [logits];
        },
    };
    return { generator: new Generator(hyperparameters, model), contexts };
};
test("generate() produces outputLength tokens, showing and passing over an end token", () => {
    const { generator, contexts } = generatorChoosing([THE, tokenizer.endOfTextId, THE]);
    assert.equal(generator.generate("Once"), `Once the${BPETokenizer.END_OF_TEXT} the the the`);
    assert.equal(contexts.length, 5);
});
test("generateUntilEnd() stops generating at the end token, and leaves it out", () => {
    const { generator, contexts } = generatorChoosing([THE, THE, tokenizer.endOfTextId, THE]);
    assert.equal(generator.generateUntilEnd("Once", 20), "Once the the");
    // The model is not run again after choosing the end token.
    assert.equal(contexts.length, 3);
});
test("generateUntilEnd() returns only the prompt if the end token comes first", () => {
    const { generator, contexts } = generatorChoosing([tokenizer.endOfTextId]);
    assert.equal(generator.generateUntilEnd("Once", 20), "Once");
    assert.equal(contexts.length, 1);
});
test("generateUntilEnd() stops at its maximum if there is no end token, whatever outputLength is", () => {
    const { generator, contexts } = generatorChoosing([THE], 2);
    assert.equal(generator.generateUntilEnd("Once", 4), "Once the the the the");
    assert.equal(contexts.length, 4);
});
test("generateUntilEnd() feeds the model only the most recent tokens that fit in its context", () => {
    const { generator, contexts } = generatorChoosing([THE], 5, 3);
    const prompt = tokenizer.encode("One two three four");
    generator.generateUntilEnd("One two three four", 2);
    assert.deepEqual(contexts[0], prompt.slice(-3));
    assert.deepEqual(contexts[1], [...prompt.slice(-2), THE]);
});
test("generateUntilEnd() rejects an empty prompt", () => {
    const { generator } = generatorChoosing([THE]);
    assert.throws(() => generator.generateUntilEnd("", 5), /at least one token/);
});
