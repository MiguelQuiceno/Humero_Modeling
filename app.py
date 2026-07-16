"""
Visualización interactiva (tipo GeoGebra) de la transformación afín
T(x) = (sR)x + b entre dos pares esfera/recta intersectante.

Ejecutar:  python app.py   ->  http://localhost:8050
"""

import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, ctx

from geometria import analizar_par, construir_T, T_animada, rotacion_interpolada

# ---------------------------------------------------------------- utilidades

def malla_esfera(C, r, nu=32, nv=24):
    u = np.linspace(0, 2 * np.pi, nu)
    v = np.linspace(0, np.pi, nv)
    x = C[0] + r * np.outer(np.cos(u), np.sin(v))
    y = C[1] + r * np.outer(np.sin(u), np.sin(v))
    z = C[2] + r * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def puntos_esfera(C, r, nu=32, nv=24):
    """Nube (N,3) de puntos de la superficie, para poder aplicarles T."""
    x, y, z = malla_esfera(C, r, nu, nv)
    return np.column_stack([x.ravel(), y.ravel(), z.ravel()]), x.shape


def segmento_recta(par, largo=None):
    """Segmento de la recta centrado en el pie de la perpendicular Q."""
    if largo is None:
        largo = 3.0 * par["r"]
    t = np.linspace(-largo, largo, 2)
    return par["Q"] + np.outer(t, par["v"])


def trazas_par(par, color, nombre, opacidad=0.55, dash=None):
    """Trazas Plotly (esfera + recta + centro + pts de intersección) de un par."""
    xs, ys, zs = malla_esfera(par["C"], par["r"])
    seg = segmento_recta(par)
    tr = [
        go.Surface(x=xs, y=ys, z=zs, opacity=opacidad, showscale=False,
                   colorscale=[[0, color], [1, color]],
                   name=nombre, hoverinfo="name"),
        go.Scatter3d(x=seg[:, 0], y=seg[:, 1], z=seg[:, 2], mode="lines",
                     line=dict(color=color, width=7, dash=dash),
                     name=f"L de {nombre}"),
        go.Scatter3d(x=[par["C"][0]], y=[par["C"][1]], z=[par["C"][2]],
                     mode="markers", marker=dict(size=4, color=color),
                     name=f"C de {nombre}", showlegend=False),
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


# ---------------------------------------------------------------- app

app = Dash(__name__)
app.title = "T(x) = sRx + b — esferas y rectas"

def num(id_, val, step=0.1, w="70px"):
    return dcc.Input(id=id_, type="number", value=val, step=step,
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
           "Arrastra para rotar la vista, usa el deslizador o ▶ para animar."),

    html.Div([
        html.Div([
            fila_par("p1", "Par 1 (origen — húmero 1)",
                     [0, 0, 0], 1.0, [0.4, 0, -2], [0, 0, 1], "#1f77b4"),
            fila_par("p2", "Par 2 (destino — húmero 2)",
                     [5, 3, 1], 2.0, [5.5, 1, 1], [0, 1, 0.3], "#2ca02c"),
            html.Div([
                html.Button("▶ Animar", id="play", n_clicks=0,
                            style={"marginRight": "8px"}),
                html.Button("⟲ Reiniciar", id="reset", n_clicks=0),
            ], style={"margin": "6px 0"}),
            dcc.Slider(id="t", min=0, max=1, step=0.02, value=0,
                       marks={0: "t=0 (par 1)", 1: "t=1 (par 2)"},
                       tooltip={"placement": "bottom"}),
            dcc.Interval(id="tick", interval=60, disabled=True),
            html.Div(id="panel-T", style={"marginTop": "10px"}),
        ], style={"flex": "0 0 430px", "paddingRight": "14px"}),

        dcc.Graph(id="grafica", style={"flex": "1", "height": "82vh"}),
    ], style={"display": "flex"}),
])


def leer_pares(vals):
    (c10, c11, c12, r1, p10, p11, p12, d10, d11, d12,
     c20, c21, c22, r2, p20, p21, p22, d20, d21, d22) = vals
    par1 = analizar_par([c10, c11, c12], r1, [p10, p11, p12], [d10, d11, d12])
    par2 = analizar_par([c20, c21, c22], r2, [p20, p21, p22], [d20, d21, d22])
    return par1, par2


IDS = ([f"p1-c{i}" for i in range(3)] + ["p1-r"] +
       [f"p1-p{i}" for i in range(3)] + [f"p1-d{i}" for i in range(3)] +
       [f"p2-c{i}" for i in range(3)] + ["p2-r"] +
       [f"p2-p{i}" for i in range(3)] + [f"p2-d{i}" for i in range(3)])


@app.callback(
    Output("grafica", "figure"),
    Output("panel-T", "children"),
    Input("t", "value"),
    *[Input(i, "value") for i in IDS],
)
def actualizar(t, *vals):
    if any(v is None for v in vals):
        return go.Figure(), html.Div("Completa todos los campos.")
    par1, par2 = leer_pares(vals)
    trans = construir_T(par1, par2)

    fig = go.Figure()
    # Par 1 (origen) y Par 2 (destino), siempre visibles como referencia
    for tr in trazas_par(par1, "#1f77b4", "Par 1", opacidad=0.25):
        fig.add_trace(tr)
    for tr in trazas_par(par2, "#2ca02c", "Par 2", opacidad=0.25):
        fig.add_trace(tr)

    if trans["valido"]:
        # Copia animada del par 1 bajo T_t
        pts, forma = puntos_esfera(par1["C"], par1["r"])
        pts_t = T_animada(pts, t, par1, par2, trans)
        xs = pts_t[:, 0].reshape(forma)
        ys = pts_t[:, 1].reshape(forma)
        zs = pts_t[:, 2].reshape(forma)
        fig.add_trace(go.Surface(x=xs, y=ys, z=zs, opacity=0.85, showscale=False,
                                 colorscale=[[0, "#ff7f0e"], [1, "#ff7f0e"]],
                                 name="T_t(esfera 1)"))
        seg = segmento_recta(par1)
        seg_t = T_animada(seg, t, par1, par2, trans)
        fig.add_trace(go.Scatter3d(x=seg_t[:, 0], y=seg_t[:, 1], z=seg_t[:, 2],
                                   mode="lines",
                                   line=dict(color="#ff7f0e", width=8),
                                   name="T_t(L1)"))

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(aspectmode="data"),
        legend=dict(x=0.01, y=0.99),
        uirevision="mantener-camara",   # conserva la vista al animar
    )

    # ------- panel de información -------
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


@app.callback(
    Output("tick", "disabled"),
    Output("play", "children"),
    Input("play", "n_clicks"),
    State("tick", "disabled"),
    prevent_initial_call=True,
)
def alternar_play(_, deshabilitado):
    encender = deshabilitado
    return (not encender), ("⏸ Pausar" if encender else "▶ Animar")


@app.callback(
    Output("t", "value"),
    Input("tick", "n_intervals"),
    Input("reset", "n_clicks"),
    State("t", "value"),
    prevent_initial_call=True,
)
def avanzar(_, __, t):
    if ctx.triggered_id == "reset":
        return 0
    t = (t or 0) + 0.02
    return 0 if t > 1.0001 else round(t, 4)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)
