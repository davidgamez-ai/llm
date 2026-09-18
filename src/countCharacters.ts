import * as fs from "fs";
import * as path from "path";

const filePath = path.join(import.meta.dirname, "..", "data", "the-verdict.txt");
const text = fs.readFileSync(filePath, "utf-8");

console.log(`Total number of characters: ${text.length}`);
