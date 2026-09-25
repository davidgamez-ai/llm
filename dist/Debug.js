/** Switches that control debug log output in different parts of the model */
const DEBUG = {
    /** Switches on log output for the embedding */
    EMBEDDING: false,
    /** Switches on log output for the Query, Key and Value matrices */
    QKV: false,
    /** Switches on log output for the attention weight calculation */
    ATTENTION: false,
    /** Switches on log output for the context vector calculation */
    CONTEXT: false
};
export default DEBUG;
