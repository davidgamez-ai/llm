/**
 * mulberry32: a small, fast seeded PRNG. Deterministically derives a
 * stream of 32-bit states from the seed and scrambles each one via
 * Math.imul/xorshift into a uniform value in [0, 1), so weight matrices can
 * be built reproducibly for a given seed. Math.random() cannot be seeded, so
 * it cannot be used when reproducible results are needed.
 *
 * @param seed - Initial 32-bit PRNG state.
 * @returns A function that, on each call, advances the PRNG state and returns the next pseudo-random value in [0, 1).
 */
function createSeededRandom(seed) {
    let state = seed;
    return () => {
        // Advance the state with a fixed odd increment (32-bit overflow wraps via `| 0`).
        state = (state + 0x6d2b79f5) | 0;
        // Scramble the state (xorshift + multiply) so consecutive states don't correlate.
        let t = Math.imul(state ^ (state >>> 15), 1 | state);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}
export default createSeededRandom;
