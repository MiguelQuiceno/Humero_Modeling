# Transformación afín T(x) = (sR)x + b entre pares esfera/recta

Visualización interactiva (tipo GeoGebra, en el navegador) de la transformación afín que convierte un par esfera+recta intersectante en otro — convertir un "húmero" en el otro mediante la razón s = r₂/r₁.

## Ejecutar en GitHub Codespaces

```bash
pip install -r requirements.txt
python app.py
```

Codespaces detecta el puerto **8050** y muestra una notificación "Open in Browser" (o pestaña **PORTS** → puerto 8050 → 🌐). La app corre en `http://localhost:8050`.

## Qué hace

- Dibuja los dos pares esfera/recta (Par 1 azul, Par 2 verde) con sus puntos de intersección recta-esfera marcados con ✕.
- Construye para cada par la base ortonormal derecha `M_i = [w_i, v_i, n_i]`:
  - `w_i`: unitario de C_i hacia su proyección ortogonal sobre L_i
  - `v_i`: dirección unitaria de L_i
  - `n_i = w_i × v_i`
- Calcula `s = r₂/r₁`, `R = M₂M₁ᵀ`, `b = C₂ − sRC₁` y **verifica que R es ortogonal** (‖RRᵀ−I‖ ≈ 0, det R = +1).
- **Anima** la copia naranja del Par 1 transformándose en el Par 2 con `T_t(x) = s^t · R_t · (x − C₁) + C_t` (slerp para la rotación), donde T₀ = identidad y T₁ = T. Botón ▶ o deslizador t.
- Todos los parámetros (C, r, P, d de cada par) son editables en vivo.
- **La razón s se muestra en rojo cuando la conversión no es posible**:
  - alguna recta no intersecta su esfera, o
  - alguna recta pasa por el centro (w indefinido → no hay base), o
  - R resulta no ortogonal numéricamente.

## Archivos

- `geometria.py` — la matemática: bases, T, verificación de ortogonalidad, interpolación de la animación.
- `app.py` — la interfaz interactiva (Dash + Plotly 3D).

## Nota

El signo de `v_i` (dirección de la recta) es arbitrario; invertir `d` de un par produce una R distinta pero igualmente ortogonal — la esfera y la recta terminan igual, cambia la "orientación" del giro.
