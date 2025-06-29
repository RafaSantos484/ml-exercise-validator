export default class Utils {
  static async sleep(ms: number) {
    await new Promise((resolve) => {
      setTimeout(() => {
        resolve(null);
      }, ms);
    });
  }

  static translate(str: string) {
    const translationTable: Record<string, string> = {
      incorrect: "incorreto",
      correct: "correto",
    };

    return translationTable[str] ?? str;
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
