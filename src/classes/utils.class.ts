export default class Utils {
  static async sleep(ms: number) {
    await new Promise((resolve) => {
      setTimeout(() => {
        resolve(null);
      }, ms);
    });
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
}
