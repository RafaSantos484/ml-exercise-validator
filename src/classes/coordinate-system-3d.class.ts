import Point3d from "./point3d.class";

export default class CoordinateSystem3D {
  origin: Point3d;
  xDir: Point3d;
  yDir: Point3d;
  zDir: Point3d;

  static canonicalSystem = new CoordinateSystem3D(
    new Point3d([0, 0, 0]),
    new Point3d([1, 0, 0]),
    new Point3d([0, 1, 0])
  );

  constructor(origin: Point3d, xDir: Point3d, yDir: Point3d) {
    this.origin = origin;
    this.xDir = xDir;
    this.yDir = yDir;
    this.zDir = this.xDir.cross(this.yDir);
  }

  toLocal(point: Point3d): Point3d {
    // Vector from point to the origin of the system
    const relative = point.subtract(this.origin);

    // Projection onto local system axes
    const xLocal = relative.dot(this.xDir.normalize());
    const yLocal = relative.dot(this.yDir.normalize());
    const zLocal = relative.dot(this.zDir.normalize());

    return new Point3d([
      xLocal / this.xDir.norm(),
      yLocal / this.yDir.norm(),
      zLocal / this.zDir.norm(),
    ]);
  }
}
