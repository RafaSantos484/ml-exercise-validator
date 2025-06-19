export default class Utils {
  static async sleep(ms: number) {
    await new Promise((resolve) => {
      setTimeout(() => {
        resolve(null);
      }, ms);
    });
  }

  static async readJson<T>(path: string) {
    const response = await fetch(path);
    return (await response.json()) as T;
  }

  static getCombinations<T>(list: T[], n: number): T[][] {
    if (n === 0) return [[]];
    if (list.length < n) return [];

    const [first, ...remainder] = list;
    const withFirst = this.getCombinations(remainder, n - 1).map((c) => [
      first,
      ...c,
    ]);
    const withoutFirst = this.getCombinations(remainder, n);

    return [...withFirst, ...withoutFirst];
  }

  static translate(str: string) {
    const translationTable: Record<string, string> = {
      incorrect: "incorreto",
      correct: "correto",
    };

    return translationTable[str] ?? str;
  }

  static minkowskiDistance(a: number[], b: number[], p: number): number {
    if (a.length !== b.length) {
      throw new Error("Vectors must be of same length");
    }

    const sum = a.reduce(
      (acc, val, i) => acc + Math.pow(Math.abs(val - b[i]), p),
      0
    );
    return Math.pow(sum, 1 / p);
  }

  static sigmoid(z: number): number {
    return 1 / (1 + Math.exp(-z));
  }

  static softmax(logits: number[]): number[] {
    const max = Math.max(...logits); // evita overflow
    const exps = logits.map((z) => Math.exp(z - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map((e) => e / sum);
  }
}
