"""
Visualización interactiva (tipo GeoGebra) de la transformación afín
T(x) = (sR)x + b entre dos pares esfera/recta intersectante.

La animación se precalcula como frames de Plotly y corre 100% en el
navegador (sin viajes al servidor), por lo que es fluida en Codespaces.

Ejecutar:  python app.py   ->  puerto 8050
"""

import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

from geometria import analizar_par, construir_T, T_animada

N_FRAMES = 31          # fotogramas de la animación (t = 0 .. 1)
NU, NV = 24, 16        # resolución de la malla de la esfera (ligera)

# ---------------------------------------------------------------- utilidades

def malla_esfera(C, r, nu=NU, nv=NV):
    u = np.linspace(0, 2 * np.pi, nu)
    v = np.linspace(0, np.pi, nv)
    x = C[0] + r * np.outer(np.cos(u), np.sin(v))
    y = C[1] + r * np.outer(np.sin(u), np.sin(v))
    z = C[2] + r * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def puntos_esfera(C, r):
    x, y, z = malla_esfera(C, r)
    return np.column_stack([x.ravel(), y.ravel(), z.ravel()]), x.shape


def segmento_recta(par, largo=None):
    if largo is None:
        largo = 3.0 * par["r"]
    t = np.linspace(-largo, largo, 2)
    return par["Q"] + np.outer(t, par["v"])


def trazas_par(par, color, nombre, opacidad):
    xs, ys, zs = malla_esfera(par["C"], par["r"])
    seg = segmento_recta(par)
    tr = [
        go.Surface(x=xs, y=ys, z=zs, opacity=opacidad, showscale=False,
                   colorscale=[[0, color], [1, color]],
                   name=nombre, hoverinfo="name"),
        go.Scatter3d(x=seg[:, 0], y=seg[:, 1], z=seg[:, 2], mode="lines",
                     line=dict(color=color, width=7), name=f"L de {nombre}"),
    ]
    if par["inter_pts"]:
        pts = np.array(par["inter_pts"])
        tr.append(go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                               mode="markers",
                               marker=dict(size=5, color="black", symbol="x"),
                               name="intersección", showlegend=False))
    return tr


def fmt_mat(R):
    if R is None:
        return "—"
    return html.Pre("\n".join("  ".join(f"{v: .4f}" for v in fila) for fila in R),
                    style={"margin": 0, "fontSize": "13px"})


def figura(par1, par2, trans):
    """Figura completa con frames precalculados de la animación."""
    base = []
    base += trazas_par(par1, "#1f77b4", "Par 1", 0.25)
    base += trazas_par(par2, "#2ca02c", "Par 2", 0.25)

    fig = go.Figure()
    frames = []

    if trans["valido"]:
        pts, forma = puntos_esfera(par1["C"], par1["r"])
        seg = segmento_recta(par1)

        def trazas_movil(t):
            p = T_animada(pts, t, par1, par2, trans)
            sg = T_animada(seg, t, par1, par2, trans)
            return [
                go.Surface(x=p[:, 0].reshape(forma), y=p[:, 1].reshape(forma),
                           z=p[:, 2].reshape(forma), opacity=0.9, showscale=False,
                           colorscale=[[0, "#ff7f0e"], [1, "#ff7f0e"]],
                           name="T_t(esfera 1)", hoverinfo="name"),
                go.Scatter3d(x=sg[:, 0], y=sg[:, 1], z=sg[:, 2], mode="lines",
                             line=dict(color="#ff7f0e", width=8), name="T_t(L1)"),
            ]

        ts = np.linspace(0, 1, N_FRAMES)
        movil0 = trazas_movil(0.0)
        # Las 2 trazas móviles van PRIMERO para que los frames (que solo
        # redibujan traces=[0,1]) las actualicen en el navegador.
        for tr in movil0 + base:
            fig.add_trace(tr)
        frames = [go.Frame(data=trazas_movil(t), traces=[0, 1],
                           name=f"{t:.2f}") for t in ts]
        fig.frames = frames

        pasos = [dict(method="animate", label=f"{t:.2f}",
                      args=[[f"{t:.2f}"],
                            dict(mode="immediate",
                                 frame=dict(duration=0, redraw=True),
                                 transition=dict(duration=0))])
                 for t in ts]
        fig.update_layout(
            updatemenus=[dict(
                type="buttons", direction="left",
                x=0.02, y=0.02, xanchor="left", yanchor="bottom",
                buttons=[
                    dict(label="▶ Animar", method="animate",
                         args=[None, dict(mode="immediate", fromcurrent=True,
                                          frame=dict(duration=50, redraw=True),
                                          transition=dict(duration=0))]),
                    dict(label="⏸ Pausar", method="animate",
                         args=[[None], dict(mode="immediate",
                                            frame=dict(duration=0, redraw=False),
                                            transition=dict(duration=0))]),
                ])],
            sliders=[dict(steps=pasos, x=0.02, y=0.08, len=0.5,
                          currentvalue=dict(prefix="t = "))],
        )
    else:
        for tr in base:
            fig.add_trace(tr)

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(aspectmode="data"),
        legend=dict(x=0.01, y=0.99),
        uirevision="mantener-camara",
    )
    return fig


# ---------------------------------------------------------------- app

app = Dash(__name__)
app.title = "T(x) = sRx + b — esferas y rectas"

def num(id_, val, step=0.1, w="70px"):
    return dcc.Input(id=id_, type="number", value=val, step=step,
                     debounce=True,       # actualiza al salir del campo / Enter
                     style={"width": w, "marginRight": "4px"})

def fila_par(pref, titulo, C, r, P, d, color):
    return html.Div([
        html.H4(titulo, style={"color": color, "margin": "8px 0 4px"}),
        html.Div(["Centro C: ", *[num(f"{pref}-c{i}", C[i]) for i in range(3)],
                  "  radio r: ", num(f"{pref}-r", r, w="60px")]),
        html.Div(["Punto P de L: ", *[num(f"{pref}-p{i}", P[i]) for i in range(3)],
                  "  dirección d: ", *[num(f"{pref}-d{i}", d[i]) for i in range(3)]],
                 style={"marginTop": "4px"}),
    ], style={"padding": "8px", "border": f"2px solid {color}",
              "borderRadius": "8px", "marginBottom": "8px"})

app.layout = html.Div([
    html.H2("Transformación afín T(x) = (sR)x + b entre dos pares esfera/recta"),
    html.P("Convertir un “húmero” en el otro mediante la razón s = r₂/r₁. "
           "Edita un valor y pulsa Enter (o sal del campo) para actualizar. "
           "▶ Animar y el deslizador t están dentro de la gráfica."),
    html.Div([
        html.Div([
            fila_par("p1", "Par 1 (origen — húmero 1)",
                     [0, 0, 0], 1.0, [0.4, 0, -2], [0, 0, 1], "#1f77b4"),
            fila_par("p2", "Par 2 (destino — húmero 2)",
                     [5, 3, 1], 2.0, [5.5, 1, 1], [0, 1, 0.3], "#2ca02c"),
            dcc.Loading(html.Div(id="panel-T"), type="dot"),
        ], style={"flex": "0 0 430px", "paddingRight": "14px"}),
        dcc.Graph(id="grafica", style={"flex": "1", "height": "85vh"}),
    ], style={"display": "flex"}),
])

IDS = ([f"p1-c{i}" for i in range(3)] + ["p1-r"] +
       [f"p1-p{i}" for i in range(3)] + [f"p1-d{i}" for i in range(3)] +
       [f"p2-c{i}" for i in range(3)] + ["p2-r"] +
       [f"p2-p{i}" for i in range(3)] + [f"p2-d{i}" for i in range(3)])


@app.callback(
    Output("grafica", "figure"),
    Output("panel-T", "children"),
    *[Input(i, "value") for i in IDS],
)
def actualizar(*vals):
    if any(v is None for v in vals):
        return go.Figure(), html.Div("Completa todos los campos.")
    (c10, c11, c12, r1, p10, p11, p12, d10, d11, d12,
     c20, c21, c22, r2, p20, p21, p22, d20, d21, d22) = vals
    par1 = analizar_par([c10, c11, c12], r1, [p10, p11, p12], [d10, d11, d12])
    par2 = analizar_par([c20, c21, c22], r2, [p20, p21, p22], [d20, d21, d22])
    trans = construir_T(par1, par2)
    fig = figura(par1, par2, trans)

    ok = trans["valido"]
    color_razon = "#111" if ok else "#d62728"   # ROJO cuando no se puede
    panel = html.Div([
        html.H4("Transformación T", style={"margin": "4px 0"}),
        html.Div([
            html.B("Razón (scaling) s = r₂/r₁ = "),
            html.Span(f"{trans['s']:.4f}",
                      style={"color": color_razon, "fontWeight": "bold",
                             "fontSize": "20px"}),
            html.Span("  ✔ conversión posible" if ok
                      else "  ✘ NO se puede convertir",
                      style={"color": "#2ca02c" if ok else "#d62728",
                             "fontWeight": "bold"}),
        ]),
        html.Div([html.B("R (rotación) = "), fmt_mat(trans["R"])]),
        html.Div([html.B("b (traslación) = "),
                  "—" if trans["b"] is None else
                  "( " + ", ".join(f"{v:.4f}" for v in trans["b"]) + " )"]),
        html.Div([
            html.B("Verificación de ortogonalidad: "),
            ("—" if trans["err_orto"] is None else
             f"‖RRᵀ − I‖ = {trans['err_orto']:.2e},  det(R) = {trans['det']:.6f}  "
             + ("→ R es ortogonal ✔" if trans["ortogonal"] else "→ NO ortogonal ✘")),
        ], style={"marginTop": "4px"}),
        html.Ul([html.Li(p, style={"color": "#d62728"})
                 for p in trans["problemas"]]) if trans["problemas"] else None,
    ], style={"border": "1px solid #ccc", "borderRadius": "8px",
              "padding": "10px", "background": "#fafafa"})
    return fig, panel


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)
