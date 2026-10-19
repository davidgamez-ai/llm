import { encode, decode, vocabularySize, EndOfText } from "gpt-tokenizer/encoding/r50k_base";

/**
 * A tokenizer backed by the pre-trained byte pair encoding (BPE) vocabulary
 * that gpt-tokenizer ships (OpenAI's r50k_base encoding, which has the same
 * vocabulary as GPT-2 and was also used by GPT-3's original models such as
 * davinci). Unlike Tokenizer, there is no vocabulary to build, so there is
 * no train() method.
 *
 * The <|endoftext|> special token is allowed in the input text and encoded
 * as its own id, as the book does with allowed_special={"<|endoftext|>"},
 * so several documents can be joined into one training text.
 */
class BPETokenizer {
  /** The text of the special token that marks the boundary between documents. */
  static readonly END_OF_TEXT: string = EndOfText;

  /**
   * The id of END_OF_TEXT (50256 in r50k_base). Looked up from the library
   * rather than hard-coded. Encoding the token on its own is reliable: the
   * library only fails to find special tokens part way through the text
   * (see encode()).
   */
  private static readonly END_OF_TEXT_ID: number = BPETokenizer.lookUpEndOfTextId();

  private readonly _vocabularySize: number = vocabularySize;

  /**
   * Finds the id of END_OF_TEXT by encoding it on its own.
   *
   * @returns The id of END_OF_TEXT.
   * @throws Error if the token does not encode to exactly one id, which would mean the library has changed.
   */
  private static lookUpEndOfTextId(): number {
    const ids = encode(EndOfText, { allowedSpecial: new Set([EndOfText]) });
    if (ids.length !== 1) {
      throw new Error(`BPETokenizer| Expected ${EndOfText} to encode to 1 id; got ${JSON.stringify(ids)}`);
    }
    return ids[0];
  }

  /** The number of unique tokens in the r50k_base vocabulary, including END_OF_TEXT. */
  get vocabularySize(): number {
    return this._vocabularySize;
  }

  /** The id that END_OF_TEXT is encoded as. */
  get endOfTextId(): number {
    return BPETokenizer.END_OF_TEXT_ID;
  }

  /**
   * Converts text into token ids. Never fails on unseen text: BPE falls back
   * to byte-level tokens for anything outside its vocabulary. Each
   * occurrence of END_OF_TEXT in the text is encoded as the single id
   * endOfTextId rather than as the characters "<", "|", "endoftext" and so on.
   *
   * gpt-tokenizer 4.0.0 is meant to do this itself when given
   * allowedSpecial, but its search for special tokens uses a sticky regular
   * expression that only matches at the exact position the search starts.
   * As a result it only recognises a special token at the very start of the
   * text, and silently encodes any later one as ordinary text. To avoid
   * this, the text is split at each END_OF_TEXT here, each piece is encoded
   * separately, and endOfTextId is placed between the pieces. This gives the
   * same result as tiktoken, which also encodes the text between special
   * tokens separately.
   *
   * @param text - The text to encode. May contain END_OF_TEXT.
   * @returns The token ids, in order.
   */
  encode(text: string): number[] {
    const pieces: string[] = text.split(EndOfText);

    // Encode each piece, inserting END_OF_TEXT_ID before every piece except the first.
    const ids: number[] = [];
    for (let i = 0; i < pieces.length; i++) {
      if (i > 0) {
        ids.push(BPETokenizer.END_OF_TEXT_ID);
      }
      ids.push(...encode(pieces[i]));
    }
    return ids;
  }

  /**
   * Converts token ids back to text. endOfTextId is decoded as END_OF_TEXT.
   *
   * @param ids - The token ids to decode.
   * @returns The decoded text.
   */
  decode(ids: number[]): string {
    return decode(ids);
  }
}

export default BPETokenizer;
