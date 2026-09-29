/** Switches that control debug log output in different parts of the model */
const DEBUG = {
    /** Switches on log output for the embedding */
    EMBEDDING: true,

    /** Switches on log output for the Query, Key and Value matrices */
    QKV: false,

    /** Switches on log output for the attention weight calculation */
    ATTENTION: false,

    /** Switches on log output for the context vector calculation */
    CONTEXT: false,

    /** Switches on log output for the feed forward network */
    FEED_FORWARD: false,

    /** Switches on log output for the transformer block */
    TRANSFORMER: false,

    /** Switches on log output for the GPT model */
    GPT_MODEL: false,

    /** Switches on log output for the linear output layer */
    LINEAR_OUTPUT_LAYER: false,

    /** Enables verbose log output, including tables of weights, etc. */
    VERBOSE: false
};

export default DEBUG;
