import { test } from "node:test";
import assert from "node:assert/strict";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";
import { readCsvColumn } from "../CsvReader.js";
/*
 * Writes text to a new CSV file in a temporary folder and returns its path,
 * so the tests do not depend on the data folder.
 */
const writeCsv = (text) => {
    const folder = fs.mkdtempSync(path.join(os.tmpdir(), "csv-test-"));
    const filePath = path.join(folder, "data.csv");
    fs.writeFileSync(filePath, text, "utf-8");
    return filePath;
};
// A small file in the same format as TinyStories: one quoted "text" column, with line breaks and escaped quotes in the stories.
const STORIES_CSV = 'text\n' +
    '"Once upon a time, there was a cat."\n' +
    '"Tom said, ""Hello!""\n\nThen he left."\n' +
    '"The end, with a comma."\n';
const STORIES = [
    "Once upon a time, there was a cat.",
    'Tom said, "Hello!"\n\nThen he left.',
    "The end, with a comma.",
];
test("readCsvColumn() reads quoted values containing commas, line breaks and escaped quotes", () => {
    assert.deepEqual(readCsvColumn(writeCsv(STORIES_CSV), "text", 10), STORIES);
});
test("readCsvColumn() stops after maximumValues values", () => {
    assert.deepEqual(readCsvColumn(writeCsv(STORIES_CSV), "text", 2), STORIES.slice(0, 2));
});
test("readCsvColumn() returns every value when the file has fewer than requested", () => {
    assert.equal(readCsvColumn(writeCsv(STORIES_CSV), "text", 1000).length, 3);
});
test("readCsvColumn() gives the same values whatever the chunk size, including one byte at a time", () => {
    const filePath = writeCsv(STORIES_CSV);
    for (const chunkSize of [1, 2, 3, 7, 64]) {
        assert.deepEqual(readCsvColumn(filePath, "text", 10, 0, chunkSize), STORIES, `chunk size ${chunkSize}`);
    }
});
test("readCsvColumn() skips the first skipValues values", () => {
    const filePath = writeCsv(STORIES_CSV);
    assert.deepEqual(readCsvColumn(filePath, "text", 10, 1), STORIES.slice(1));
    assert.deepEqual(readCsvColumn(filePath, "text", 1, 1), [STORIES[1]]);
    // Skipping works whatever the chunk size, including when a skipped value is split between chunks.
    assert.deepEqual(readCsvColumn(filePath, "text", 10, 2, 3), [STORIES[2]]);
});
test("readCsvColumn() returns no values when skipValues reaches the end of the file", () => {
    assert.deepEqual(readCsvColumn(writeCsv(STORIES_CSV), "text", 10, 3), []);
    assert.deepEqual(readCsvColumn(writeCsv(STORIES_CSV), "text", 10, 50), []);
});
test("readCsvColumn() keeps multi-byte characters intact when they are split between chunks", () => {
    // Curly quotes, an accented letter and an emoji take 3, 2 and 4 bytes in UTF-8.
    const story = "“Hello,” said Zoë 😀";
    const filePath = writeCsv(`text\n"${story}"\n`);
    assert.deepEqual(readCsvColumn(filePath, "text", 10, 0, 1), [story]);
});
test("readCsvColumn() reads the named column from a file with several, unquoted values and \\r\\n line breaks", () => {
    const filePath = writeCsv('id,text,length\r\n1,First story,11\r\n2,"Second, story",13\r\n');
    assert.deepEqual(readCsvColumn(filePath, "text", 10), ["First story", "Second, story"]);
    assert.deepEqual(readCsvColumn(filePath, "id", 10), ["1", "2"]);
});
test("readCsvColumn() skips empty lines and reads a last record with no line break", () => {
    const filePath = writeCsv('text\n\n"First"\n\n"Last"');
    assert.deepEqual(readCsvColumn(filePath, "text", 10), ["First", "Last"]);
});
test("readCsvColumn() throws a clear error for a missing column or an empty file", () => {
    assert.throws(() => readCsvColumn(writeCsv(STORIES_CSV), "story", 10), /no column "story"/);
    assert.throws(() => readCsvColumn(writeCsv(""), "text", 10), /no header/);
});
