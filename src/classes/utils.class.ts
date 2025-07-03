import type { ValidationResult } from "../types";

export default class Utils {
  static async sleep(ms: number) {
    await new Promise((resolve) => {
      setTimeout(() => {
        resolve(null);
      }, ms);
    });
  }

  static isBetween(value: number, min: number, max: number) {
    return value >= min && value <= max;
  }

  static translate(str: string) {
    const translationTable: Record<string, ValidationResult> = {
      incorrect: { text: "incorreto", color: "red", isCorrect: false },
      correct: { text: "correto", color: "green", isCorrect: true },
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
