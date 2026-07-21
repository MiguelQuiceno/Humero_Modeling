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


# ==================================================================
#  Parte 2: recta transformada, plano por T(L1) y L2, y boost O(3,1)
# ==================================================================

ETA = np.diag([-1.0, 1.0, 1.0, 1.0])   # métrica de Minkowski (-,+,+,+)


def _mink(a, b):
    return float(np.asarray(a) @ ETA @ np.asarray(b))


def linea_transformada(par1, trans):
    """
    Imagen de la recta L1 bajo la transformación afín T(x)=sRx+b.
    Devuelve el punto ancla T(Q1), la dirección unitaria y un segmento.
    (T manda esfera1 -> esfera2 exactamente, pero L1 -> T(L1) que en
     general NO es L2.)
    """
    if not trans["valido"]:
        return None
    sR = trans["s"] * trans["R"]
    ancla = sR @ par1["Q"] + trans["b"]
    direc = trans["R"] @ par1["v"]
    direc = direc / np.linalg.norm(direc)
    largo = 3.0 * (trans["s"] * par1["r"])
    t = np.linspace(-largo, largo, 2)
    seg = ancla + np.outer(t, direc)
    return dict(ancla=ancla, dir=direc, seg=seg)


def plano_por_lineas(linT, par2, trans):
    """
    Plano que contiene a T(L1) y a L2 (ambas rectas resultantes).
    Verifica que ese plano pasa por el centro C2 de la esfera destino.
    Devuelve normal, punto base (C2), residuos de verificación y si
    T(L1) coincide con L2.
    """
    if linT is None:
        return None
    A = linT["ancla"]            # punto de T(L1)
    uA = linT["dir"]             # dirección de T(L1)  (= v2)
    B = par2["Q"]               # punto de L2
    C2 = par2["C"]
    conect = B - A              # vector que une ambas rectas

    cr = np.cross(uA, conect)
    if np.linalg.norm(cr) > 1e-9:
        normal = cr / np.linalg.norm(cr)   # normal del plano por ambas rectas
        coinciden = False
    else:
        # Las dos rectas coinciden (T(L1)=L2): plano = C2 + span(v2,w2)
        normal = par2["n"]
        coinciden = True

    res_C2 = _mink_res(C2 - A, normal)   # C2 en el plano  (≈0)
    res_A = 0.0
    res_B = _mink_res(B - A, normal)     # L2 en el plano  (≈0)
    # T(L1)=L2  <=>  mismas anclas  <=>  beta1=beta2
    T_es_L2 = np.linalg.norm(A - B) < 1e-9

    # malla del plano para dibujar (centrada en C2)
    u1 = par2["v"]
    u2 = np.cross(normal, u1); u2 /= np.linalg.norm(u2)
    L = 3.2 * par2["r"]
    gs = np.linspace(-L, L, 2)
    P = (C2[:, None, None]
         + u1[:, None, None] * gs[None, :, None]
         + u2[:, None, None] * gs[None, None, :])
    return dict(normal=normal, C2=C2, res_C2=res_C2, res_L2=res_B,
                coinciden=coinciden, T_es_L2=T_es_L2,
                plano_x=P[0], plano_y=P[1], plano_z=P[2])


def _mink_res(vec, normal):
    return float(np.dot(vec, normal))


def cuadrivelocidad(par):
    """
    4-velocidad asociada a la recta: dirección = v (dirección de la recta),
    rapidez beta = dist/r  (subluminal  <=>  la recta corta la esfera).
    U = gamma (1, beta v),  <U,U> = -1.  Es el vector temporal de la base
    de Lorentz sobre la recta.
    """
    beta = par["dist"] / par["r"]
    valida = (beta < 1.0 - 1e-9) and (not par["degenerado"])
    beta_c = min(beta, 0.999999)
    gamma = 1.0 / np.sqrt(1.0 - beta_c ** 2)
    v = par["v"]
    U = gamma * np.array([1.0, beta_c * v[0], beta_c * v[1], beta_c * v[2]])
    return dict(U=U, beta=beta, gamma=gamma, valida=valida)


def base_lorentz(par):
    """
    Base (tétrada) de Lorentz sobre la recta: F=[U, s1, s2, s3] ortonormal
    en la métrica de Minkowski (F^T η F = η), obtenida por Gram-Schmidt de
    Minkowski a partir de la 4-velocidad U y los ejes espaciales (w,v,n).
    """
    cv = cuadrivelocidad(par)
    U = cv["U"]
    semillas = [np.concatenate([[0.0], par["w"]]),
                np.concatenate([[0.0], par["v"]]),
                np.concatenate([[0.0], par["n"]])]
    base = [U]
    for s in semillas:
        x = s.copy()
        for f in base:
            x = x - f * (_mink(f, s) / _mink(f, f))
        nrm2 = _mink(x, x)
        if nrm2 > 1e-9:
            base.append(x / np.sqrt(nrm2))
    F = np.column_stack(base) if len(base) == 4 else None
    err = np.linalg.norm(F.T @ ETA @ F - ETA) if F is not None else None
    return dict(F=F, U=U, err_orto=err, cv=cv)


def boost_lorentz(par1, par2):
    """
    Boost de Lorentz (matriz de O(3,1)) construido a partir de las bases de
    Lorentz de ambas rectas: es el boost puro en el plano temporal
    generado por U1 y U2 que lleva U1 -> U2 (equivale a la doble reflexión
    de Lorentz). Verifica pertenencia a O(3,1) y que es un boost puro
    (autovalores {e^-φ,1,1,e^+φ}).
    """
    b1 = base_lorentz(par1)
    b2 = base_lorentz(par2)
    U1, U2 = b1["U"], b2["U"]
    cv1, cv2 = b1["cv"], b2["cv"]

    problemas = []
    if not cv1["valida"]:
        problemas.append("β₁ = dist₁/r₁ ≥ 1: L1 no corta la esfera 1 "
                         "(4-velocidad no física)")
    if not cv2["valida"]:
        problemas.append("β₂ = dist₂/r₂ ≥ 1: L2 no corta la esfera 2 "
                         "(4-velocidad no física)")

    if problemas:
        return dict(valido=False, problemas=problemas, F1=b1["F"], F2=b2["F"],
                    b1=cv1, b2=cv2, L=None)

    g = -_mink(U1, U2)                     # γ relativo = cosh φ
    if g < 1.0 + 1e-12:                    # U1 == U2  ->  boost identidad
        L = np.eye(4)
        phi = 0.0
    else:
        e1 = U2 - g * U1
        e1 = e1 / np.sqrt(_mink(e1, e1))
        ch, sh = g, np.sqrt(g * g - 1.0)
        A0 = -np.outer(U1, ETA @ U1)
        A1 = np.outer(e1, ETA @ e1)
        Sw = np.outer(U1, ETA @ e1) - np.outer(e1, ETA @ U1)
        L = np.eye(4) + (ch - 1.0) * (A0 + A1) + sh * Sw
        phi = np.arccosh(g)

    err_O31 = np.linalg.norm(L.T @ ETA @ L - ETA)
    det = np.linalg.det(L)
    orthochronous = L[0, 0] >= 1.0 - 1e-9
    mapea = np.linalg.norm(L @ U1 - U2)
    eig = np.sort(np.real(np.linalg.eigvals(L)))
    # boost puro: dos autovalores =1 y un par recíproco e^{±φ}
    es_boost = (err_O31 < 1e-6 and abs(det - 1) < 1e-6 and orthochronous
                and np.sum(np.abs(eig - 1.0) < 1e-6) >= 2)

    return dict(valido=True, problemas=[], L=L, F1=b1["F"], F2=b2["F"],
                err_F1=b1["err_orto"], err_F2=b2["err_orto"],
                U1=U1, U2=U2, b1=cv1, b2=cv2,
                gamma_rel=g, rapidez=phi, beta_rel=np.tanh(phi),
                err_O31=err_O31, det=det, L00=float(L[0, 0]),
                orthochronous=orthochronous, mapea_err=mapea,
                eig=eig, es_boost=es_boost)
