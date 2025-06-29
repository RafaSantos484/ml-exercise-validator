import type { ValidationResult } from "./models/model.class";

export default class Utils {
  static async sleep(ms: number) {
    await new Promise((resolve) => {
      setTimeout(() => {
        resolve(null);
      }, ms);
    });
  }

  static translate(str: string) {
    const translationTable: Record<string, ValidationResult> = {
      incorrect: { text: "incorreto", color: "red" },
      correct: { text: "correto", color: "green" },
    };

    return translationTable[str] ?? { text: str, color: "yellow" };
  }

  static getMeanAndStdDev(arr: number[]) {
    if (arr.length === 0) {
      return { mean: 0, stdDev: 0 };
    }

    const mean = arr.reduce((sum, value) => sum + value, 0) / arr.length;
    const stdDev =
      Math.sqrt(
        arr.reduce((sum, value) => sum + Math.pow(value - mean, 2), 0) /
          arr.length
      ) || 0;

    return { mean, stdDev };
  }
}
