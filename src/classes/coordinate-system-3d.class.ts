import Point3d from "./point3d.class";

export default class CoordinateSystem3D {
  origin: Point3d;
  x_axis: Point3d;
  y_axis: Point3d;
  z_axis: Point3d;

  constructor(origin: Point3d, xDir: Point3d, yDir: Point3d) {
    this.origin = origin;
    this.x_axis = xDir.normalize();
    this.z_axis = this.x_axis.cross(yDir).normalize();
    this.y_axis = this.z_axis.cross(this.x_axis).normalize(); // Re-orthogonalize y
  }

  toLocal(point: Point3d): Point3d {
    // Vector from point to the origin of the system
    const relative = point.subtract(this.origin);

    // Projection onto local system axes
    const xLocal = relative.dot(this.x_axis);
    const yLocal = relative.dot(this.y_axis);
    const zLocal = relative.dot(this.z_axis);

    return new Point3d([xLocal, yLocal, zLocal]);
  }
}
