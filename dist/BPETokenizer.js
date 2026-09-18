import { encode, decode, vocabularySize } from "gpt-tokenizer/encoding/r50k_base";
// A tokenizer backed by the pre-trained byte pair encoding (BPE) vocabulary
// that gpt-tokenizer ships (OpenAI's r50k_base encoding, used by GPT-3's
// original models such as davinci). Unlike Tokenizer, there is no
// vocabulary to build, so there is no train() method.
class BPETokenizer {
    _vocabularySize = vocabularySize;
    // The number of unique tokens in the r50k_base vocabulary.
    get vocabularySize() {
        return this._vocabularySize;
    }
    // Delegates to gpt-tokenizer's encode(), which never fails on unseen text:
    // BPE falls back to byte-level tokens for anything outside its vocabulary.
    encode(text) {
        return encode(text);
    }
    // Delegates to gpt-tokenizer's decode() to convert token ids back to text.
    decode(ids) {
        return decode(ids);
    }
}
export default BPETokenizer;
