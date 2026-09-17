import { encode, decode } from "gpt-tokenizer";
// A tokenizer backed by the pre-trained byte pair encoding (BPE) vocabulary
// that gpt-tokenizer ships (OpenAI's o200k_base encoding, used by GPT-4o and
// newer models). Unlike Tokenizer, there is no vocabulary to build, so there
// is no train() method.
class BPETokenizer {
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
