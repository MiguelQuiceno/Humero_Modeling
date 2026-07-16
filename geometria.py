"""
Geometría del ejercicio: transformación afín T(x) = (sR)x + b
entre dos pares esfera/recta intersectante.

Para cada par (esfera C_i, r_i ; recta L_i con punto P_i y dirección d_i):
  w_i : vector unitario desde C_i hacia su proyección ortogonal sobre L_i
  v_i : vector unitario de dirección de L_i
  n_i : w_i x v_i  (completa la base derecha)
  M_i = [w_i, v_i, n_i]  (columnas)

  s = r2 / r1
  R = M2 @ M1.T
  b = C2 - s R C1
"""

import numpy as np

TOL = 1e-9


def proyeccion_sobre_recta(C, P, d):
    """Proyección ortogonal del punto C sobre la recta {P + t d}."""
    d = np.asarray(d, dtype=float)
    du = d / np.linalg.norm(d)
    return np.asarray(P, dtype=float) + np.dot(np.asarray(C, dtype=float) - P, du) * du


def analizar_par(C, r, P, d):
    """
    Analiza un par esfera/recta. Devuelve un dict con:
      w, v, n, M, dist (distancia centro-recta), intersecta (bool),
      degenerado (bool: la recta pasa por el centro, w indefinido),
      puntos de intersección recta-esfera (si existen).
    """
    C = np.asarray(C, dtype=float)
    P = np.asarray(P, dtype=float)
    d = np.asarray(d, dtype=float)

    v = d / np.linalg.norm(d)
    Q = proyeccion_sobre_recta(C, P, v)      # pie de la perpendicular
    diff = Q - C
    dist = np.linalg.norm(diff)

    degenerado = dist < 1e-8                 # recta pasa por el centro -> w indefinido
    intersecta = dist <= r + TOL

    w = diff / dist if not degenerado else np.zeros(3)
    n = np.cross(w, v)
    M = np.column_stack([w, v, n])

    # Intersección recta-esfera: |P + t v - C|^2 = r^2
    pts = []
    if intersecta and not np.isnan(dist):
        h2 = r * r - dist * dist
        if h2 >= -TOL:
            h = np.sqrt(max(h2, 0.0))
            pts = [Q - h * v, Q + h * v]

    return dict(C=C, r=r, P=P, v=v, w=w, n=n, M=M, Q=Q,
                dist=dist, intersecta=intersecta,
                degenerado=degenerado, inter_pts=pts)


def construir_T(par1, par2):
    """
    Construye la transformación T(x) = sR x + b entre los dos pares.
    Devuelve dict con s, R, b, verificación de ortogonalidad y validez.
    """
    problemas = []
    if not par1["intersecta"]:
        problemas.append("La recta L1 NO intersecta la esfera 1")
    if not par2["intersecta"]:
        problemas.append("La recta L2 NO intersecta la esfera 2")
    if par1["degenerado"]:
        problemas.append("L1 pasa por el centro C1: w1 indefinido (no hay base)")
    if par2["degenerado"]:
        problemas.append("L2 pasa por el centro C2: w2 indefinido (no hay base)")

    s = par2["r"] / par1["r"]

    if problemas:
        return dict(valido=False, s=s, R=None, b=None,
                    err_orto=None, det=None, ortogonal=False,
                    problemas=problemas)

    M1, M2 = par1["M"], par2["M"]
    R = M2 @ M1.T
    b = par2["C"] - s * R @ par1["C"]

    # Verificación de ortogonalidad: R R^T = I y det(R) = +1
    err_orto = np.linalg.norm(R @ R.T - np.eye(3))
    det = np.linalg.det(R)
    ortogonal = err_orto < 1e-6 and abs(det - 1.0) < 1e-6
    if not ortogonal:
        problemas.append(f"R no es ortogonal (||RRᵀ−I||={err_orto:.2e}, det={det:.4f})")

    return dict(valido=ortogonal, s=s, R=R, b=b,
                err_orto=err_orto, det=det, ortogonal=ortogonal,
                problemas=problemas)


def rotacion_interpolada(R, t):
    """Interpola entre I (t=0) y R (t=1) vía eje-ángulo (slerp)."""
    from scipy.spatial.transform import Rotation
    rv = Rotation.from_matrix(R).as_rotvec()
    return Rotation.from_rotvec(t * rv).as_matrix()


def T_animada(x, t, par1, par2, trans):
    """
    Transformación intermedia T_t tal que T_0 = identidad y T_1 = T.
      T_t(x) = s(t) R(t) (x - C1) + C(t)
    con s(t) = s^t (interp. geométrica), R(t) slerp, C(t) lineal C1->C2.
    En t=1: s R (x - C1) + C2 = sRx + (C2 - sR C1) = sRx + b = T(x).
    """
    s_t = trans["s"] ** t
    R_t = rotacion_interpolada(trans["R"], t)
    C_t = (1 - t) * par1["C"] + t * par2["C"]
    x = np.asarray(x, dtype=float)
    return (s_t * (R_t @ (x - par1["C"]).T)).T + C_t
