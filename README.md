# Esferas, rectas, transformación afín, plano y boost de Lorentz O(3,1)

App interactiva (Dash + Plotly, tipo GeoGebra) con **sliders en tiempo real**.

## Qué muestra
Dos pares esfera + recta (la recta corta la esfera). Se construye:

1. **Transformación afín** `T(x) = (sR)x + b` con `s = r2/r1`,
   `R = M2 M1^T`, `b = C2 − sR·C1`, donde `Mi = [wi, vi, ni]` es la base
   ortonormal derecha de cada recta (`w`: del centro a la recta; `v`:
   dirección de la recta; `n = w×v`). Verifica que `R` es ortogonal.
   La razón `s` sale en rojo cuando la conversión no es posible.

2. **Recta transformada `T(L1)`** (naranja). `T` manda esfera1→esfera2
   *exactamente*, pero L1 va a `T(L1)`, que en general **no** es L2:
   son paralelas y `T(L1)=L2` solo si `dist1/r1 = dist2/r2` (β1=β2).

3. **Plano por `T(L1)` y `L2`** (gris). Se verifica numéricamente que
   ese plano **pasa por el centro `C2`** de la esfera destino
   (es el plano por `C2` con normal `n2`); `C2` aparece marcado.

4. **Boost de Lorentz `Λ ∈ O(3,1)`** construido desde **bases de Lorentz
   sobre las rectas**. A cada recta se le asocia la 4-velocidad
   `U = γ(1, β v̂)` con `β = dist/r` (subluminal ⇔ la recta corta la
   esfera). `Λ` es el boost puro en el plano temporal `span(U1,U2)` que
   lleva `U1 → U2`. Se verifica `Λ^T η Λ = η`, `det Λ = 1`, `Λ⁰₀ ≥ 1`,
   y que es boost puro (autovalores `{e^-φ, 1, 1, e^+φ}`, rapidez `φ`).

Marca la casilla **"Ver animación del morfismo"** para ver la
transformación afín animada (corre en el navegador, botón ▶).

## Ejecutar en GitHub Codespaces
```bash
pip install -r requirements.txt
python app.py
```
En la pestaña **PORTS**, junto al puerto **8050**, pulsa el globo 🌐
("Open in Browser").

## Archivos
- `geometria.py` — toda la matemática (afín, recta transformada, plano,
  4-velocidades, bases de Lorentz, boost O(3,1)); verificada numéricamente.
- `app.py` — interfaz con sliders en tiempo real.
- `requirements.txt` — numpy, scipy, dash, plotly.

## Nota
El signo de `v` (dirección de la recta) es arbitrario, así que hay más de
una `R`/boost válidos (todos ortogonales / en O(3,1)).
