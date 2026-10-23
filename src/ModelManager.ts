import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import GPTModel, { NamedParameter } from "./GPTModel.js";
import Hyperparameters, { HyperparameterValues } from "./Hyperparameters.js";

/** A safetensors file's header, and where its data section starts. */
interface SafetensorsHeader {
  /** The parsed JSON header: an entry for each array, plus the optional "__metadata__" of strings. */
  header: Record<string, unknown>;

  /** The position in the file of the first byte of the data section. */
  dataStart: number;

  /** The number of bytes in the data section. */
  dataLength: number;
}

/**
 * Called by ModelManager after each parameter array is written or read, so
 * that progress can be shown while a large model is saved or loaded.
 *
 * @param bytesDone - The number of bytes of weights written or read so far.
 * @param bytesTotal - The total number of bytes of weights to write or read.
 */
export type ProgressCallback = (bytesDone: number, bytesTotal: number) => void;

/** The entry for one parameter array in a safetensors header. */
interface TensorEntry {
  /** The number format: always "F32" (32-bit float) here. */
  dtype: string;

  /** The shape of the array, e.g. [rows, columns]. */
  shape: number[];

  /** The array's first byte and one past its last byte, counted from the start of the data section. */
  data_offsets: [number, number];
}

/**
 * Saves a GPTModel's weights to a file and loads them back, in the
 * safetensors format (https://github.com/huggingface/safetensors), so the
 * files can also be read in Python by safetensors, PyTorch or NumPy.
 *
 * A safetensors file has three parts:
 *
 *     [8 bytes]  the length of the header, as a little-endian unsigned 64-bit integer
 *     [header]   JSON giving each array's name, number format, shape and
 *                position in the data section, plus optional string metadata
 *     [data]     the raw bytes of every array, back to back
 *
 * Every parameter is stored as 4-byte floats, exactly as the model holds it
 * in memory, so nothing is lost and the file is 4 bytes per parameter (about
 * 650 MB for GPT-2 small). A text format such as JSON would be several times
 * larger and, at GPT-2 small's size, longer than the longest string
 * JavaScript allows.
 *
 * The model's hyperparameters are saved with the weights, as a JSON string
 * in the header's "__metadata__" (safetensors only allows strings there).
 * loadModel() uses them to build a model of the right shape before loading
 * its weights; load() uses them to check that a model it is given matches
 * the file. The optimizer's running averages are not saved, so training
 * resumed from a file starts AdamW afresh.
 */
class ModelManager {
  /**
   * The default location of OpenAI's GPT-2 small weights for loadGPT2():
   * gpt2-small.safetensors in the project's models folder. Found from this
   * compiled file's location (dist/), so it works however the program is
   * launched.
   */
  static readonly GPT2_SMALL_FILE: string = path.join(import.meta.dirname, "..", "models", "gpt2-small.safetensors");

  /** Where the GPT-2 small weights can be downloaded from, for error messages. */
  static readonly GPT2_SMALL_URL: string = "https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors";

  /** The key in the header's "__metadata__" under which the hyperparameters are saved. */
  private static readonly HYPERPARAMETERS_KEY: string = "hyperparameters";

  /**
   * The hyperparameters that decide the shape of the model's arrays, which
   * load() checks against the file's. The others (learning rate, dropout
   * and so on) only affect training, so they may differ.
   */
  private static readonly ARCHITECTURE: (keyof HyperparameterValues)[] = [
    "embeddingSize",
    "contextLength",
    "numberAttentionHeads",
    "numberTransformerBlocks",
    "qkvBias",
    "feedForwardBias",
  ];

  /** The safetensors name for 32-bit floats. */
  private static readonly DTYPE: string = "F32";

  /** The number of bytes in a 32-bit float. */
  private static readonly BYTES_PER_VALUE: number = 4;

  /** The number of bytes in the header length at the start of the file. */
  private static readonly HEADER_LENGTH_BYTES: number = 8;

  /**
   * Checks that this computer stores numbers little-endian, as safetensors
   * requires. Float32Array uses the computer's own byte order, so the bytes
   * are written and read as they are, without conversion. Almost every
   * computer (x86, ARM) is little-endian.
   *
   * @throws Error if the computer is big-endian.
   */
  private static checkEndianness(): void {
    if (os.endianness() !== "LE") {
      throw new Error("ModelManager| safetensors files are little-endian; this computer is big-endian");
    }
  }

  /**
   * Views a parameter array's memory as bytes, without copying it, so it
   * can be written to or read from a file directly.
   *
   * @param values - The parameter array.
   * @returns The same memory as a Uint8Array.
   */
  private static bytesOf(values: Float32Array): Uint8Array {
    return new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
  }

  /**
   * Saves every trainable parameter in a model, and the hyperparameters it
   * was built from, to a safetensors file, replacing the file if it exists.
   *
   * The arrays are written one at a time straight from the model's memory,
   * so no second copy of the weights is made.
   *
   * The file is first written under a temporary name in the same folder
   * (fileName + ".tmp"), then renamed to fileName. If saving fails part way,
   * the temporary file is deleted, so an existing file at fileName is never
   * left half overwritten.
   *
   * @param model - The model whose weights are saved.
   * @param fileName - The path of the file to write, conventionally ending in ".safetensors".
   * @param onProgress - Optional; called after each parameter array is written.
   * @throws Error if the computer is big-endian, or if the file cannot be written.
   */
  save(model: GPTModel, fileName: string, onProgress?: ProgressCallback): void {
    ModelManager.checkEndianness();
    const parameters: NamedParameter[] = model.getParameters();

    // Build the header: each array's position in the data section follows on from the previous one's.
    const header: Record<string, TensorEntry | Record<string, string>> = {
      __metadata__: {
        format: "pt",
        parameterCount: String(model.getParameterCount()),
        [ModelManager.HYPERPARAMETERS_KEY]: JSON.stringify(model.hyperparameters),
      },
    };
    let offset = 0;
    for (const { name, shape, values } of parameters) {
      header[name] = { dtype: ModelManager.DTYPE, shape, data_offsets: [offset, offset + values.byteLength] };
      offset += values.byteLength;
    }

    /*
     * The header is padded with spaces to a multiple of 8 bytes, as the
     * safetensors library does, so the data section starts on an 8-byte
     * boundary and other readers can view it directly as floats.
     */
    let headerBytes: Buffer = Buffer.from(JSON.stringify(header), "utf-8");
    const padding = (ModelManager.HEADER_LENGTH_BYTES - (headerBytes.length % ModelManager.HEADER_LENGTH_BYTES)) % ModelManager.HEADER_LENGTH_BYTES;
    headerBytes = Buffer.concat([headerBytes, Buffer.alloc(padding, " ")]);

    const headerLength = Buffer.alloc(ModelManager.HEADER_LENGTH_BYTES);
    headerLength.writeBigUInt64LE(BigInt(headerBytes.length));

    const temporaryFileName = fileName + ".tmp";
    const file = fs.openSync(temporaryFileName, "w");
    try {
      try {
        fs.writeSync(file, headerLength);
        fs.writeSync(file, headerBytes);
        let bytesWritten = 0;
        for (const { values } of parameters) {
          fs.writeSync(file, ModelManager.bytesOf(values));
          bytesWritten += values.byteLength;
          onProgress?.(bytesWritten, offset);
        }
      } finally {
        fs.closeSync(file);
      }
      fs.renameSync(temporaryFileName, fileName);
    } catch (error) {
      fs.rmSync(temporaryFileName, { force: true });
      throw error;
    }
  }

  /**
   * Creates a model from a safetensors file written by save(): builds a
   * GPTModel from the hyperparameters saved in the file, then loads the
   * file's weights into it.
   *
   * The saved seed is restored, but the random number stream starts from
   * its beginning, not from where it had reached when the model was saved.
   *
   * @param fileName - The path of the safetensors file to read.
   * @param mode - The mode to put the model in, since it is usually loaded for a different purpose from the one it was saved in. Defaults to GENERATION_MODE.
   * @param onProgress - Optional; called after each parameter array is read.
   * @returns The new model, with the saved weights.
   * @throws Error if the file cannot be read, is not a valid safetensors file or has no saved hyperparameters.
   * @throws RangeError if a saved hyperparameter is invalid, or the file's arrays do not match its hyperparameters.
   */
  loadModel(fileName: string, mode: number = Hyperparameters.GENERATION_MODE, onProgress?: ProgressCallback): GPTModel {
    const file = fs.openSync(fileName, "r");
    let values: HyperparameterValues | undefined;
    try {
      values = ModelManager.savedHyperparameters(ModelManager.readHeader(file, fileName).header, fileName);
    } finally {
      fs.closeSync(file);
    }
    if (values === undefined) {
      throw new Error(`ModelManager| ${fileName} has no saved hyperparameters, so a model cannot be built from it; create one and use load()`);
    }

    const hyperparameters = Hyperparameters.fromJSON(values);
    hyperparameters.mode = mode;
    const model = new GPTModel(hyperparameters);
    this.load(model, fileName, onProgress);
    return model;
  }

  /**
   * Loads the weights in a safetensors file into a model, replacing its
   * current weights. The model must have been created with the same
   * hyperparameters as the model that was saved: if the file has saved
   * hyperparameters, those that decide the model's shape must match the
   * model's; and every parameter array must be in the file with the same
   * shape, and the file must contain nothing else. Hyperparameters that only
   * affect training, such as the learning rate, may differ, and the model
   * keeps its own.
   *
   * The values are copied into the model's existing arrays rather than
   * replacing them, because other objects hold on to those arrays: AdamW
   * keys its running averages by them, for example. Each array is read
   * straight from the file into the model's memory, so the whole file is
   * never held in memory at once.
   *
   * Everything is checked before any values are copied, so if the file does
   * not match the model, the model is left unchanged.
   *
   * @param model - The model to load the weights into, changed in place.
   * @param fileName - The path of the safetensors file to read.
   * @param onProgress - Optional; called after each parameter array is read.
   * @throws Error if the computer is big-endian, or if the file cannot be read or is not a valid safetensors file.
   * @throws RangeError if the file's saved hyperparameters do not match the model's, or its arrays do not: a missing or extra array, a different shape or a number format other than F32.
   */
  load(model: GPTModel, fileName: string, onProgress?: ProgressCallback): void {
    ModelManager.checkEndianness();
    const parameters: NamedParameter[] = model.getParameters();

    const file = fs.openSync(fileName, "r");
    try {
      const { header, dataStart, dataLength } = ModelManager.readHeader(file, fileName);

      // Comparing the hyperparameters first gives a clearer error than the shape checks would.
      const saved = ModelManager.savedHyperparameters(header, fileName);
      if (saved !== undefined) {
        const current = model.hyperparameters.toJSON();
        for (const name of ModelManager.ARCHITECTURE) {
          if (saved[name] !== current[name]) {
            throw new RangeError(`ModelManager| ${fileName} was saved from a model with ${name} ${saved[name]}; this model has ${current[name]}`);
          }
        }
      }

      const entries = ModelManager.matchEntries(header, parameters, fileName, dataLength);

      // Everything matches, so copy each array from the file into the model.
      const bytesTotal = parameters.reduce((sum, { values }) => sum + values.byteLength, 0);
      let bytesRead = 0;
      parameters.forEach(({ values }, i) => {
        ModelManager.readFully(file, ModelManager.bytesOf(values), dataStart + entries[i].data_offsets[0], fileName);
        bytesRead += values.byteLength;
        onProgress?.(bytesRead, bytesTotal);
      });
    } finally {
      fs.closeSync(file);
    }
  }

  /**
   * Creates a GPT-2 small model with OpenAI's pretrained weights, read from
   * the safetensors file published on Hugging Face (see GPT2_SMALL_URL).
   *
   * The model is built from the default Hyperparameters, which are GPT-2
   * small's: embeddingSize 768, contextLength 1024, 12 heads, 12 blocks and
   * biases everywhere. Its layer normalization (epsilon 1e-5), GELU (tanh
   * approximation) and attention scaling (by the square root of the head
   * size) match GPT-2's, so with these weights it calculates what GPT-2
   * does.
   *
   * Hugging Face's arrays are named and arranged differently from this
   * model's, so they are mapped across rather than loaded with load():
   *
   *     wte                 → token embedding, and the linear output layer
   *     wpe                 → position embedding
   *     h.N.ln_1, h.N.ln_2  → block N's attention and feed forward layer normalizations
   *     h.N.attn.c_attn     → block N's query, key and value weights and biases, split between the heads
   *     h.N.attn.c_proj     → block N's attention output projection
   *     h.N.mlp.c_fc        → block N's feed forward first layer
   *     h.N.mlp.c_proj      → block N's feed forward second layer
   *     ln_f                → final layer normalization
   *
   * Hugging Face stores each weight matrix as [inputs, outputs], as this
   * model does, so no matrix needs transposing. GPT-2 uses the token
   * embedding as its output layer too ("weight tying"); this model has a
   * separate output layer, so wte is copied into both, which is why the
   * model has 163 million parameters rather than GPT-2's 124 million. The
   * h.N.attn.bias arrays in the file are GPT-2's causal masks, not weights,
   * and are ignored.
   *
   * c_attn holds the query, key and value weights side by side in one
   * [768, 2304] matrix: columns 0-767 are the queries, 768-1535 the keys and
   * 1536-2303 the values. Within each, head h has the 64 columns starting
   * at h x 64, which is the order this model concatenates its heads in.
   *
   * Building the model first fills it with random weights, which takes a
   * few seconds, before they are replaced.
   *
   * @param fileName - The path of the GPT-2 small safetensors file. Defaults to GPT2_SMALL_FILE (models/gpt2-small.safetensors).
   * @param mode - The mode to put the model in. Defaults to GENERATION_MODE.
   * @param onProgress - Optional; called after each of the file's arrays is read.
   * @returns The new model, with GPT-2 small's weights.
   * @throws Error if the file does not exist (the message says how to download it), cannot be read or is not a valid safetensors file, or if the computer is big-endian.
   * @throws RangeError if an array the model needs is missing from the file, or has a different shape or number format.
   */
  loadGPT2(fileName: string = ModelManager.GPT2_SMALL_FILE, mode: number = Hyperparameters.GENERATION_MODE, onProgress?: ProgressCallback): GPTModel {
    if (!fs.existsSync(fileName)) {
      throw new Error(
        `ModelManager| The GPT-2 small weights were not found at ${fileName}. ` +
        `Download them from ${ModelManager.GPT2_SMALL_URL} and save them there, e.g. with: ` +
        `curl.exe -L ${ModelManager.GPT2_SMALL_URL} -o "${fileName}"`
      );
    }
    ModelManager.checkEndianness();

    const file = fs.openSync(fileName, "r");
    try {
      const { header, dataStart, dataLength } = ModelManager.readHeader(file, fileName);

      // GPT-2 small's hyperparameters are the defaults.
      const hyperparameters = new Hyperparameters();
      const { embeddingSize, numberTransformerBlocks, weightMatrixColumns } = hyperparameters;

      /*
       * Find and check every array before building the model, so that a
       * wrong file is reported straight away rather than after the few
       * seconds it takes to build the model.
       */
      const entry = (name: string, shape: number[]): TensorEntry =>
        ModelManager.checkEntry(header, name, shape, fileName, dataLength);
      const blockNames = ["ln_1.weight", "ln_1.bias", "attn.c_attn.weight", "attn.c_attn.bias", "attn.c_proj.weight", "attn.c_proj.bias",
        "ln_2.weight", "ln_2.bias", "mlp.c_fc.weight", "mlp.c_fc.bias", "mlp.c_proj.weight", "mlp.c_proj.bias"];
      const hiddenSize = 4 * embeddingSize;
      const blockShapes: Record<string, number[]> = {
        "ln_1.weight": [embeddingSize], "ln_1.bias": [embeddingSize],
        "attn.c_attn.weight": [embeddingSize, 3 * embeddingSize], "attn.c_attn.bias": [3 * embeddingSize],
        "attn.c_proj.weight": [embeddingSize, embeddingSize], "attn.c_proj.bias": [embeddingSize],
        "ln_2.weight": [embeddingSize], "ln_2.bias": [embeddingSize],
        "mlp.c_fc.weight": [embeddingSize, hiddenSize], "mlp.c_fc.bias": [hiddenSize],
        "mlp.c_proj.weight": [hiddenSize, embeddingSize], "mlp.c_proj.bias": [embeddingSize],
      };
      const entries = new Map<string, TensorEntry>();
      entries.set("wte.weight", entry("wte.weight", [hyperparameters.vocabularySize, embeddingSize]));
      entries.set("wpe.weight", entry("wpe.weight", [hyperparameters.contextLength, embeddingSize]));
      for (let b = 0; b < numberTransformerBlocks; b++) {
        for (const name of blockNames) {
          entries.set(`h.${b}.${name}`, entry(`h.${b}.${name}`, blockShapes[name]));
        }
      }
      entries.set("ln_f.weight", entry("ln_f.weight", [embeddingSize]));
      entries.set("ln_f.bias", entry("ln_f.bias", [embeddingSize]));

      hyperparameters.mode = mode;
      const model = new GPTModel(hyperparameters);

      // Reads one of the file's arrays straight into a Float32Array (a new one if none is given), reporting progress.
      const bytesTotal = [...entries.values()].reduce((sum, { data_offsets: [start, end] }) => sum + end - start, 0);
      let bytesRead = 0;
      const read = (name: string, into?: Float32Array): Float32Array => {
        const [start, end] = entries.get(name)!.data_offsets;
        const values = into ?? new Float32Array((end - start) / ModelManager.BYTES_PER_VALUE);
        ModelManager.readFully(file, ModelManager.bytesOf(values), dataStart + start, fileName);
        bytesRead += end - start;
        onProgress?.(bytesRead, bytesTotal);
        return values;
      };

      read("wte.weight", model.embedding.textEmbeddingMatrix.data);
      // GPT-2 ties its output layer to the token embedding; this model's output layer is separate, so it gets a copy.
      model.linearOutputLayer.weights.data.set(model.embedding.textEmbeddingMatrix.data);
      read("wpe.weight", model.embedding.positionEmbeddingMatrix.data);

      model.transformers.forEach((block, b) => {
        const prefix = `h.${b}.`;
        read(prefix + "ln_1.weight", block.attentionLayerNormalization.scale);
        read(prefix + "ln_1.bias", block.attentionLayerNormalization.shift);

        /*
         * Split c_attn between the heads. Each row of the weights (one per
         * input value) holds the queries, keys and values side by side;
         * head h's 64 columns of each are copied into the same row of its
         * own query, key and value matrices. The biases are one row, split
         * the same way.
         */
        const attentionWeights = read(prefix + "attn.c_attn.weight");
        const attentionBiases = read(prefix + "attn.c_attn.bias");
        block.multiHeadAttention.attentionHeads.forEach((head, h) => {
          [head.queryWeights, head.keyWeights, head.valueWeights].forEach((weights, part) => {
            const firstColumn = part * embeddingSize + h * weightMatrixColumns;
            for (let row = 0; row < embeddingSize; row++) {
              const rowStart = row * 3 * embeddingSize + firstColumn;
              weights.data.set(attentionWeights.subarray(rowStart, rowStart + weightMatrixColumns), row * weightMatrixColumns);
            }
          });
          [head.queryBiases, head.keyBiases, head.valueBiases].forEach((biases, part) => {
            const firstColumn = part * embeddingSize + h * weightMatrixColumns;
            biases.set(attentionBiases.subarray(firstColumn, firstColumn + weightMatrixColumns));
          });
        });

        read(prefix + "attn.c_proj.weight", block.multiHeadAttention.outputProjectionWeights.data);
        read(prefix + "attn.c_proj.bias", block.multiHeadAttention.outputProjectionBiases);
        read(prefix + "ln_2.weight", block.feedForwardLayerNormalization.scale);
        read(prefix + "ln_2.bias", block.feedForwardLayerNormalization.shift);
        read(prefix + "mlp.c_fc.weight", block.feedForward.firstLayerWeights.data);
        read(prefix + "mlp.c_fc.bias", block.feedForward.firstLayerBiases);
        read(prefix + "mlp.c_proj.weight", block.feedForward.secondLayerWeights.data);
        read(prefix + "mlp.c_proj.bias", block.feedForward.secondLayerBiases);
      });

      read("ln_f.weight", model.finalLayerNormalization.scale);
      read("ln_f.bias", model.finalLayerNormalization.shift);
      return model;
    } finally {
      fs.closeSync(file);
    }
  }

  /**
   * Finds one array in a file's header and checks that it is what is
   * expected: 32-bit floats, of the given shape, within the data section.
   *
   * @param header - The parsed header of the file.
   * @param name - The array's name in the file.
   * @param shape - The shape it should have.
   * @param fileName - The file's path, for error messages.
   * @param dataLength - The number of bytes in the file's data section.
   * @returns The array's header entry.
   * @throws RangeError if the array is missing, has a different shape or number format, or extends past the end of the file.
   */
  private static checkEntry(header: Record<string, unknown>, name: string, shape: number[], fileName: string, dataLength: number): TensorEntry {
    const entry = header[name] as TensorEntry | undefined;
    if (entry === undefined) {
      throw new RangeError(`ModelManager| ${fileName} has no array called ${name}: is it the GPT-2 small safetensors file?`);
    }
    if (entry.dtype !== ModelManager.DTYPE) {
      throw new RangeError(`ModelManager| ${name} in ${fileName} is ${entry.dtype}; expected ${ModelManager.DTYPE}`);
    }
    if (!Array.isArray(entry.shape) || entry.shape.length !== shape.length || entry.shape.some((size, i) => size !== shape[i])) {
      throw new RangeError(`ModelManager| ${name} in ${fileName} has shape [${entry.shape}]; expected [${shape}]`);
    }
    const [start, end] = entry.data_offsets;
    const size = shape.reduce((product, length) => product * length, 1);
    if (end - start !== size * ModelManager.BYTES_PER_VALUE || start < 0 || end > dataLength) {
      throw new RangeError(`ModelManager| ${name} in ${fileName} has invalid data offsets [${start}, ${end}]`);
    }
    return entry;
  }

  /**
   * Reads and parses the header of an open safetensors file.
   *
   * @param file - The open file.
   * @param fileName - The file's path, for error messages.
   * @returns The parsed header, and the position and length of the data section.
   * @throws Error if the file is too short, or its header is not valid JSON.
   */
  private static readHeader(file: number, fileName: string): SafetensorsHeader {
    const fileSize: number = fs.fstatSync(file).size;

    // Read the header length, then the header itself.
    const headerLengthBytes = Buffer.alloc(ModelManager.HEADER_LENGTH_BYTES);
    ModelManager.readFully(file, headerLengthBytes, 0, fileName);
    const headerLength = Number(headerLengthBytes.readBigUInt64LE());
    const dataStart: number = ModelManager.HEADER_LENGTH_BYTES + headerLength;
    if (dataStart > fileSize) {
      throw new Error(`ModelManager| ${fileName} is not a valid safetensors file: its header is longer than the file`);
    }
    const headerBytes = Buffer.alloc(headerLength);
    ModelManager.readFully(file, headerBytes, ModelManager.HEADER_LENGTH_BYTES, fileName);

    try {
      return { header: JSON.parse(headerBytes.toString("utf-8")), dataStart, dataLength: fileSize - dataStart };
    } catch {
      throw new Error(`ModelManager| ${fileName} is not a valid safetensors file: its header is not valid JSON`);
    }
  }

  /**
   * Reads the hyperparameters saved in a file's header by save().
   *
   * @param header - The parsed header of the file.
   * @param fileName - The file's path, for error messages.
   * @returns The saved hyperparameter values, or undefined if the file has none (e.g. it was written by another program).
   * @throws Error if the saved hyperparameters are not valid JSON.
   */
  private static savedHyperparameters(header: Record<string, unknown>, fileName: string): HyperparameterValues | undefined {
    const metadata = header.__metadata__ as Record<string, string> | undefined;
    const text = metadata?.[ModelManager.HYPERPARAMETERS_KEY];
    if (text === undefined) {
      return undefined;
    }
    try {
      return JSON.parse(text);
    } catch {
      throw new Error(`ModelManager| The hyperparameters saved in ${fileName} are not valid JSON`);
    }
  }

  /**
   * Checks that a file's header describes exactly the model's parameter
   * arrays, and finds the entry for each one.
   *
   * @param header - The parsed header of the file.
   * @param parameters - The model's parameter arrays.
   * @param fileName - The file's path, for error messages.
   * @param dataLength - The number of bytes in the file's data section.
   * @returns The header entry for each parameter, in the same order as parameters.
   * @throws RangeError if an array is missing, has a different shape or number format, extends past the end of the file, or the file contains arrays the model does not have.
   */
  private static matchEntries(header: Record<string, unknown>, parameters: NamedParameter[], fileName: string, dataLength: number): TensorEntry[] {
    const entries: TensorEntry[] = parameters.map(({ name, shape, values }) => {
      const entry = header[name] as TensorEntry | undefined;
      if (entry === undefined) {
        throw new RangeError(`ModelManager| ${fileName} has no values for ${name}: was it saved from a model with different hyperparameters?`);
      }
      if (entry.dtype !== ModelManager.DTYPE) {
        throw new RangeError(`ModelManager| ${name} in ${fileName} is ${entry.dtype}; expected ${ModelManager.DTYPE}`);
      }
      if (!Array.isArray(entry.shape) || entry.shape.length !== shape.length || entry.shape.some((size, i) => size !== shape[i])) {
        throw new RangeError(
          `ModelManager| ${name} in ${fileName} has shape [${entry.shape}]; the model's is [${shape}]: ` +
          `was it saved from a model with different hyperparameters?`
        );
      }
      const [start, end] = entry.data_offsets;
      if (end - start !== values.length * ModelManager.BYTES_PER_VALUE || start < 0 || end > dataLength) {
        throw new RangeError(`ModelManager| ${name} in ${fileName} has invalid data offsets [${start}, ${end}]`);
      }
      return entry;
    });

    // Extra arrays mean the file came from a different model, e.g. one with more transformer blocks.
    const names = new Set(parameters.map(({ name }) => name));
    const extra = Object.keys(header).filter((name) => name !== "__metadata__" && !names.has(name));
    if (extra.length > 0) {
      throw new RangeError(
        `ModelManager| ${fileName} has values the model does not, starting with ${extra[0]}: ` +
        `was it saved from a model with different hyperparameters?`
      );
    }
    return entries;
  }

  /**
   * Fills a buffer from a file, starting at a given position. readSync may
   * read fewer bytes than asked for, so it is called until the buffer is
   * full.
   *
   * @param file - The open file.
   * @param buffer - The buffer to fill.
   * @param position - The position in the file to start reading from.
   * @param fileName - The file's path, for error messages.
   * @throws Error if the file ends before the buffer is full.
   */
  private static readFully(file: number, buffer: Uint8Array, position: number, fileName: string): void {
    let bytesRead = 0;
    while (bytesRead < buffer.length) {
      const count = fs.readSync(file, buffer, bytesRead, buffer.length - bytesRead, position + bytesRead);
      if (count === 0) {
        throw new Error(`ModelManager| ${fileName} ended unexpectedly: it is not a complete safetensors file`);
      }
      bytesRead += count;
    }
  }
}

export default ModelManager;
