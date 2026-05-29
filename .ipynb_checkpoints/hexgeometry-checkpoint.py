import math
from hexgrid import HexCoord
from typing import Literal, Tuple


def signed_turn_angle(
    a: HexCoord,
    b: HexCoord,
    c: HexCoord,
    *,
    hex_size: float = 1.0,
    degrees_out: bool = False,
) -> float:
    """
    Signed path turn angle at B for the path A -> B -> C.

    Convention
    ----------
    - 0      : straight
    - > 0    : left turn (counterclockwise in XY)
    - < 0    : right turn (clockwise in XY)
    - range  : [-pi, pi]

    Notes
    -----
    This is NOT the same as angle_abc(...), which returns the interior angle ∠ABC.
    Here we use the forward path directions AB and BC, so straight motion gives 0.
    """
    if a == b or b == c:
        raise ValueError("Turn angle undefined when A == B or B == C.")

    ax, ay = axial_to_xy(a, hex_size=hex_size)
    bx, by = axial_to_xy(b, hex_size=hex_size)
    cx, cy = axial_to_xy(c, hex_size=hex_size)

    # Forward path vectors
    ux, uy = (bx - ax), (by - ay)   # AB
    vx, vy = (cx - bx), (cy - by)   # BC

    nu = math.hypot(ux, uy)
    nv = math.hypot(vx, vy)
    if nu == 0.0 or nv == 0.0:
        raise ValueError("Turn angle undefined due to zero-length segment.")

    # Signed angle from u -> v
    cross = ux * vy - uy * vx
    dot = ux * vx + uy * vy
    theta = math.atan2(cross, dot)   # [-pi, pi], 0 = straight

    return math.degrees(theta) if degrees_out else theta


def unsigned_turn_angle(
    a: HexCoord,
    b: HexCoord,
    c: HexCoord,
    *,
    hex_size: float = 1.0,
    degrees_out: bool = False,
) -> float:
    """
    Unsigned path turn angle at B for A -> B -> C.

    Equivalent to:
        pi - angle_abc(a,b,c)
    because angle_abc returns the interior angle ∠ABC.
    """
    interior = angle_abc(a, b, c, hex_size=hex_size, degrees_out=False)
    turn = math.pi - interior
    # Clamp tiny negative due to floating point
    if turn < 0.0 and abs(turn) < 1e-12:
        turn = 0.0
    return math.degrees(turn) if degrees_out else turn


def curvature_from_turn_chord(
    a: HexCoord,
    b: HexCoord,
    c: HexCoord,
    *,
    hex_size: float = 1.0,
    signed: bool = False,
    chord: Literal["ab", "bc", "ac", "avg_adjacent"] = "avg_adjacent",
    eps: float = 1e-12,
) -> float:
    """
    Curvature approximation using heading change + chord length:

        kappa = 2 * sin(delta / 2) / d

    where delta is the path turn angle at B (0 for straight), and d is a chosen
    chord/segment length.

    Parameters
    ----------
    signed : bool
        If True, attach the sign from the turn direction (left positive, right negative).
    chord : {"ab", "bc", "ac", "avg_adjacent"}
        Which geometric length to use as d:
          - "ab": length of segment A->B
          - "bc": length of segment B->C
          - "ac": chord from A to C
          - "avg_adjacent": 0.5 * (|AB| + |BC|)   [recommended default]

    Notes
    -----
    - For equal-length single-step paths on your hex grid, AB and BC will be the same.
    - "avg_adjacent" is usually the cleanest when segment lengths may differ.
    """
    delta_signed = signed_turn_angle(a, b, c, hex_size=hex_size, degrees_out=False)
    delta = abs(delta_signed)

    if delta <= eps:
        return 0.0

    ab = euclid_center_distance(a, b, hex_size=hex_size)
    bc = euclid_center_distance(b, c, hex_size=hex_size)
    ac = euclid_center_distance(a, c, hex_size=hex_size)

    if chord == "ab":
        d = ab
    elif chord == "bc":
        d = bc
    elif chord == "ac":
        d = ac
    elif chord == "avg_adjacent":
        d = 0.5 * (ab + bc)
    else:
        raise ValueError(f"Unsupported chord={chord!r}")

    if d <= eps:
        raise ValueError("Curvature undefined due to zero chord length.")

    kappa = 2.0 * math.sin(delta / 2.0) / d
    if signed:
        kappa = math.copysign(kappa, delta_signed)
    return kappa


def curvature_three_points(
    a: HexCoord,
    b: HexCoord,
    c: HexCoord,
    *,
    hex_size: float = 1.0,
    signed: bool = False,
    eps: float = 1e-12,
) -> float:
    """
    Three-point geometric curvature for the path A -> B -> C using the circumcircle.

    Formula:
        kappa = 4 * Area(ABC) / (|AB| * |BC| * |AC|)

    Sign convention
    ---------------
    - positive for left turn
    - negative for right turn

    Returns
    -------
    float
        Curvature magnitude (or signed curvature if signed=True).
        Returns 0.0 for collinear points.

    Notes
    -----
    This is usually the most geometric "continuous approximation" for curvature
    of a polyline through three points.
    """
    if a == b or b == c or a == c:
        raise ValueError("Three distinct points are required for three-point curvature.")

    ax, ay = axial_to_xy(a, hex_size=hex_size)
    bx, by = axial_to_xy(b, hex_size=hex_size)
    cx, cy = axial_to_xy(c, hex_size=hex_size)

    ab = euclid_center_distance(a, b, hex_size=hex_size)
    bc = euclid_center_distance(b, c, hex_size=hex_size)
    ac = euclid_center_distance(a, c, hex_size=hex_size)

    if ab <= eps or bc <= eps or ac <= eps:
        raise ValueError("Curvature undefined due to zero-length side.")

    # 2 * signed area of triangle ABC
    cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)

    if abs(cross) <= eps:
        return 0.0

    # Since Area = |cross| / 2, then:
    #   4*Area / (ab*bc*ac) = 2*|cross| / (ab*bc*ac)
    kappa = 2.0 * abs(cross) / (ab * bc * ac)

    if signed:
        kappa = math.copysign(kappa, cross)

    return kappa


def turning_radius_three_points(
    a: HexCoord,
    b: HexCoord,
    c: HexCoord,
    *,
    hex_size: float = 1.0,
    eps: float = 1e-12,
) -> float:
    """
    Radius of curvature for A -> B -> C, using the circumcircle.

    Returns
    -------
    float
        Radius R = 1 / kappa, or math.inf if the points are collinear.
    """
    kappa = curvature_three_points(a, b, c, hex_size=hex_size, signed=False, eps=eps)
    if abs(kappa) <= eps:
        return math.inf
    return 1.0 / kappa

def euclid_center_distance(a: HexCoord, b: HexCoord, *, hex_size: float = 1.0) -> float:
    """
    Euclidean distance between the centers of two hexes (pointy-top axial grid).
    """
    ax, ay = axial_to_xy(a, hex_size=hex_size)
    bx, by = axial_to_xy(b, hex_size=hex_size)
    return math.hypot(ax - bx, ay - by)

def axial_to_xy(coord: HexCoord, hex_size: float = 1.0) -> Tuple[float, float]:
    """
    Convert axial (q, r) -> 2D Euclidean (x, y) for a pointy-top hex grid.

    This matches your DIRECTIONS convention (pointy-top axial).
    Uses the common "hex radius" size parameter:
      x = sqrt(3) * size * (q + r/2)
      y = (3/2)   * size * r

    Notes
    -----
    - If your `hex_size` is not hex-radius but some other scale, that's fine:
      angles do not depend on scale.
    """
    q, r = coord.q, coord.r
    x = math.sqrt(3.0) * hex_size * (q + 0.5 * r)
    y = 1.5 * hex_size * r
    return (x, y)

def angle_abc(a: HexCoord, b: HexCoord, c: HexCoord, *,
              hex_size: float = 1.0,
              degrees_out: bool = False) -> float:
    """
    Return the geometric angle at B formed by A-B-C (i.e., ∠ABC).

    Parameters
    ----------
    a, b, c : HexCoord
        Points of the angle (A-B-C).
    hex_size : float
        Same scale you use elsewhere (e.g., grid.hex_size). Scale cancels out for angles.
    degrees_out : bool
        If True, return angle in degrees; otherwise radians.

    Returns
    -------
    float
        Angle at B in radians (default) or degrees.

    Raises
    ------
    ValueError
        If A == B or C == B (undefined angle).
    """
    if a == b or c == b:
        raise ValueError("Angle undefined when A == B or C == B.")

    ax, ay = axial_to_xy(a, hex_size=hex_size)
    bx, by = axial_to_xy(b, hex_size=hex_size)
    cx, cy = axial_to_xy(c, hex_size=hex_size)

    # Vectors BA and BC
    v1x, v1y = (ax - bx), (ay - by)   # BA
    v2x, v2y = (cx - bx), (cy - by)   # BC

    n1 = math.hypot(v1x, v1y)
    n2 = math.hypot(v2x, v2y)
    if n1 == 0.0 or n2 == 0.0:
        raise ValueError("Angle undefined due to zero-length segment (A==B or C==B).")

    dot = v1x * v2x + v1y * v2y
    cos_theta = dot / (n1 * n2)

    # Clamp for floating point safety
    cos_theta = max(-1.0, min(1.0, cos_theta))

    theta = math.acos(cos_theta)
    return math.degrees(theta) if degrees_out else theta


def hex_step_distance(hex_size: float) -> float:
    """Center-to-center distance for one axial step (all 6 neighbors)."""
    return math.sqrt(3.0) * hex_size  # pointy-top tiling

def hdist(a: "HexCoord", b: "HexCoord") -> int:
    """Axial hex Manhattan distance (# of steps)."""
    dq = abs(a.q - b.q)
    dr = abs(a.r - b.r)
    ds = abs((-a.q - a.r) - (-b.q - b.r))
    return (dq + dr + ds) // 2

def reverse_dir(direction):
    if direction is None:
        return None
    return HexCoord(-direction.q, -direction.r)