import * as fs from "fs";
import { StringDecoder } from "string_decoder";
/**
 * Reads values from one column of a CSV file, without reading more of the
 * file than it needs. This matters for large datasets such as TinyStories,
 * whose training file is nearly 2 GB: reading only the first few thousand
 * stories takes a fraction of a second, whereas loading the whole file into
 * a string would be slow and could run out of memory.
 *
 * The file is read in chunks and parsed one character at a time, following
 * the usual CSV rules (RFC 4180):
 * - The first record is the header, giving the name of each column.
 * - Records are separated by line breaks (\n or \r\n) and fields by commas.
 * - A field may be enclosed in double quotes, in which case it may contain
 *   commas and line breaks, and a double quote is written as two double
 *   quotes (""). TinyStories quotes every story, because stories contain
 *   both.
 *
 * Empty lines are skipped.
 *
 * Values can be read from part way through the file by skipping the first
 * skipValues of them. A CSV file cannot jump straight to a given record,
 * since records have different lengths, so the skipped records are still
 * read and parsed, but they are not kept. (About 12,000 TinyStories stories
 * are skipped per second.)
 *
 * @param filePath - Path to the UTF-8 CSV file.
 * @param columnName - The name of the column to read, as it appears in the header.
 * @param maximumValues - The largest number of values to read. Reading stops as soon as this many have been found.
 * @param skipValues - The number of values to skip before reading starts. Defaults to 0, to read from the first record.
 * @param chunkSize - The number of bytes read from the file at a time. Defaults to 64 KB. Only tests need to change it, to check values split across chunks.
 * @returns The column's values, in file order, starting after the skipped ones: maximumValues of them, or fewer if the file ends first.
 * @throws Error if the file has no header, or the header has no column called columnName.
 */
export function readCsvColumn(filePath, columnName, maximumValues, skipValues = 0, chunkSize = 65536) {
    const values = [];
    let columnIndex = -1;
    let skipped = 0;
    /*
     * Called with each complete record (one row, as a list of fields). The
     * first is the header, used to find the column; the next skipValues are
     * passed over; the rest supply values. Returns true once enough values
     * have been read, to stop reading.
     */
    const handleRecord = (fields) => {
        if (columnIndex === -1) {
            columnIndex = fields.indexOf(columnName);
            if (columnIndex === -1) {
                throw new Error(`CsvReader| ${filePath} has no column "${columnName}"; its columns are ${JSON.stringify(fields)}`);
            }
            return maximumValues <= 0;
        }
        if (skipped < skipValues) {
            skipped++;
            return false;
        }
        values.push(fields[columnIndex] ?? "");
        return values.length >= maximumValues;
    };
    const reader = new CsvRecordReader(handleRecord);
    const file = fs.openSync(filePath, "r");
    try {
        /*
         * A StringDecoder turns bytes into text, holding back the bytes of a
         * character that has been split between one chunk and the next until
         * the rest of it arrives. (Many characters, such as curly quotes, take
         * more than one byte in UTF-8.)
         */
        const decoder = new StringDecoder("utf8");
        const buffer = Buffer.alloc(chunkSize);
        let finished = false;
        while (!finished) {
            const bytesRead = fs.readSync(file, buffer, 0, chunkSize, null);
            if (bytesRead === 0) {
                // End of the file: complete the last record, which may not end with a line break.
                reader.add(decoder.end());
                reader.end();
                finished = true;
            }
            else {
                finished = reader.add(decoder.write(buffer.subarray(0, bytesRead)));
            }
        }
    }
    finally {
        fs.closeSync(file);
    }
    if (columnIndex === -1) {
        throw new Error(`CsvReader| ${filePath} is empty: it has no header`);
    }
    return values;
}
/**
 * Splits CSV text into records, one character at a time. Text can be added
 * in pieces of any size: a field or record that is split between pieces is
 * completed when the next piece arrives. This is a small "state machine":
 * what each character means depends on the state the reader is in (inside
 * or outside quotes, or just after a quote inside quotes).
 */
class CsvRecordReader {
    onRecord;
    /** The characters of the field being read. */
    field = "";
    /** The completed fields of the record being read. */
    fields = [];
    /** True while inside a quoted field, where commas and line breaks are part of the field. */
    inQuotes = false;
    /**
     * True just after a double quote inside a quoted field. The next character
     * decides what it was: another double quote means an escaped quote ("");
     * anything else means it closed the field.
     */
    quoteSeen = false;
    /** True once the record handler has asked for reading to stop. */
    stopped = false;
    /**
     * @param onRecord - Called with the fields of each complete record. Returns true to stop reading.
     */
    constructor(onRecord) {
        this.onRecord = onRecord;
    }
    /**
     * Processes the next piece of text.
     *
     * @param text - The next piece of the CSV text.
     * @returns True if the record handler has asked for reading to stop.
     */
    add(text) {
        for (let i = 0; i < text.length && !this.stopped; i++) {
            const character = text[i];
            if (this.quoteSeen) {
                this.quoteSeen = false;
                if (character === '"') {
                    // Two double quotes inside quotes stand for one double quote.
                    this.field += '"';
                    continue;
                }
                // The previous quote closed the field; this character is outside quotes, handled below.
                this.inQuotes = false;
            }
            if (this.inQuotes) {
                if (character === '"') {
                    this.quoteSeen = true;
                }
                else {
                    this.field += character;
                }
            }
            else if (character === '"') {
                this.inQuotes = true;
            }
            else if (character === ",") {
                this.endField();
            }
            else if (character === "\n") {
                this.endRecord();
            }
            else if (character !== "\r") {
                // \r is part of a \r\n line break, not of the field.
                this.field += character;
            }
        }
        return this.stopped;
    }
    /** Completes the last record at the end of the text, if it did not end with a line break. */
    end() {
        if (this.quoteSeen) {
            this.quoteSeen = false;
            this.inQuotes = false;
        }
        this.endRecord();
    }
    /** Completes the current field and starts a new one. */
    endField() {
        this.fields.push(this.field);
        this.field = "";
    }
    /** Completes the current record, passes it to the handler unless it is an empty line, and starts a new one. */
    endRecord() {
        if (this.stopped) {
            return;
        }
        this.endField();
        const isEmptyLine = this.fields.length === 1 && this.fields[0] === "";
        if (!isEmptyLine) {
            this.stopped = this.onRecord(this.fields);
        }
        this.fields = [];
    }
}
