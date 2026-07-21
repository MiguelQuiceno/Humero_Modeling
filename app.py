"""
Transformación afín T(x)=(sR)x+b entre dos pares esfera/recta  +
recta transformada T(L1), plano por T(L1) y L2 (que pasa por C2), y
boost de Lorentz de O(3,1) construido desde bases de Lorentz sobre las
rectas.  Interfaz interactiva tipo GeoGebra con sliders en tiempo real.

Ejecutar:  python app.py   ->  puerto 8050
"""

import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

from geometria import (analizar_par, construir_T, T_animada,
                       linea_transformada, plano_por_lineas, boost_lorentz)

# resolución ligera para que el arrastre sea fluido sobre el túnel de Codespaces
NU, NV = 20, 12
NU_A, NV_A = 16, 10
N_FRAMES = 21

AZUL, VERDE, NARANJA, GRIS = "#1f77b4", "#2ca02c", "#ff7f0e", "#888"

# ---------------------------------------------------------------- geometría→trazas

def malla_esfera(C, r, nu=NU, nv=NV):
    u = np.linspace(0, 2 * np.pi, nu)
    v = np.linspace(0, np.pi, nv)
    x = C[0] + r * np.outer(np.cos(u), np.sin(v))
    y = C[1] + r * np.outer(np.sin(u), np.sin(v))
    z = C[2] + r * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def pts_esfera(C, r, nu=NU_A, nv=NV_A):
    x, y, z = malla_esfera(C, r, nu, nv)
    return np.column_stack([x.ravel(), y.ravel(), z.ravel()]), x.shape


def seg_recta(par, largo=None):
    largo = largo if largo else 3.0 * par["r"]
    t = np.linspace(-largo, largo, 2)
    return par["Q"] + np.outer(t, par["v"])


def trazas_par(par, color, nombre, op_esfera):
    xs, ys, zs = malla_esfera(par["C"], par["r"])
    sg = seg_recta(par)
    tr = [go.Surface(x=xs, y=ys, z=zs, opacity=op_esfera, showscale=False,
                     colorscale=[[0, color], [1, color]], name=nombre,
                     hoverinfo="name"),
          go.Scatter3d(x=sg[:, 0], y=sg[:, 1], z=sg[:, 2], mode="lines",
                       line=dict(color=color, width=6), name=f"L {nombre}")]
    if par["inter_pts"]:
        p = np.array(par["inter_pts"])
        tr.append(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="markers",
                               marker=dict(size=4, color="black", symbol="x"),
                               name="corte", showlegend=False))
    tr.append(go.Scatter3d(x=[par["C"][0]], y=[par["C"][1]], z=[par["C"][2]],
                           mode="markers", marker=dict(size=3, color=color),
                           showlegend=False, hoverinfo="skip"))
    return tr


def flecha(desde, vec, color, nombre):
    """Pequeña flecha (línea + cono) para visualizar una velocidad beta*v."""
    p0 = np.asarray(desde); p1 = p0 + np.asarray(vec)
    linea = go.Scatter3d(x=[p0[0], p1[0]], y=[p0[1], p1[1]], z=[p0[2], p1[2]],
                         mode="lines", line=dict(color=color, width=5),
                         name=nombre)
    cono = go.Cone(x=[p1[0]], y=[p1[1]], z=[p1[2]],
                   u=[vec[0]], v=[vec[1]], w=[vec[2]],
                   sizemode="absolute", sizeref=0.25, showscale=False,
                   colorscale=[[0, color], [1, color]], anchor="tip",
                   hoverinfo="skip")
    return [linea, cono]


def figura(par1, par2, trans, linT, plano, boost, con_anim):
    fig = go.Figure()
    estaticas = []
    estaticas += trazas_par(par1, AZUL, "Par 1", 0.20)
    estaticas += trazas_par(par2, VERDE, "Par 2", 0.20)

    # --- recta transformada T(L1) y plano por T(L1) y L2 (pasa por C2)
    if linT is not None:
        s = linT["seg"]
        estaticas.append(go.Scatter3d(x=s[:, 0], y=s[:, 1], z=s[:, 2],
                                      mode="lines",
                                      line=dict(color=NARANJA, width=7),
                                      name="T(L1)"))
    if plano is not None:
        estaticas.append(go.Surface(x=plano["plano_x"], y=plano["plano_y"],
                                    z=plano["plano_z"], opacity=0.18,
                                    showscale=False,
                                    colorscale=[[0, GRIS], [1, GRIS]],
                                    name="plano de T(L1) y L2", hoverinfo="name"))
        C2 = plano["C2"]
        estaticas.append(go.Scatter3d(x=[C2[0]], y=[C2[1]], z=[C2[2]],
                                      mode="markers+text",
                                      marker=dict(size=6, color="crimson",
                                                  symbol="diamond"),
                                      text=["C2 en el plano"],
                                      textposition="top center",
                                      name="C2", showlegend=False))

    # --- flechas de velocidad (boost): beta1*v1 y beta2*v2 desde el origen
    if boost is not None and boost["valido"]:
        estaticas += flecha([0, 0, 0], boost["b1"]["beta"] * par1["v"], AZUL,
                            "beta1 v1")
        estaticas += flecha([0, 0, 0], boost["b2"]["beta"] * par2["v"], VERDE,
                            "beta2 v2")

    if con_anim and trans["valido"]:
        pts, forma = pts_esfera(par1["C"], par1["r"])
        sg = seg_recta(par1)

        def moviles(tt):
            p = T_animada(pts, tt, par1, par2, trans)
            l = T_animada(sg, tt, par1, par2, trans)
            return [go.Surface(x=p[:, 0].reshape(forma), y=p[:, 1].reshape(forma),
                               z=p[:, 2].reshape(forma), opacity=0.9,
                               showscale=False,
                               colorscale=[[0, NARANJA], [1, NARANJA]],
                               name="T_t(esfera1)", hoverinfo="name"),
                    go.Scatter3d(x=l[:, 0], y=l[:, 1], z=l[:, 2], mode="lines",
                                 line=dict(color="#d62728", width=8),
                                 name="T_t(L1)")]
        for tr in moviles(0.0) + estaticas:      # móviles primero (traces 0,1)
            fig.add_trace(tr)
        ts = np.linspace(0, 1, N_FRAMES)
        fig.frames = [go.Frame(data=moviles(tt), traces=[0, 1], name=f"{tt:.2f}")
                      for tt in ts]
        pasos = [dict(method="animate", label=f"{tt:.2f}",
                      args=[[f"{tt:.2f}"], dict(mode="immediate",
                                               frame=dict(duration=0, redraw=True),
                                               transition=dict(duration=0))])
                 for tt in ts]
        fig.update_layout(
            updatemenus=[dict(type="buttons", direction="left", x=0.02, y=0.05,
                              xanchor="left", yanchor="bottom", buttons=[
                dict(label="▶", method="animate",
                     args=[None, dict(mode="immediate", fromcurrent=True,
                                      frame=dict(duration=60, redraw=True),
                                      transition=dict(duration=0))]),
                dict(label="⏸", method="animate",
                     args=[[None], dict(mode="immediate",
                                        frame=dict(duration=0, redraw=False))])])],
            sliders=[dict(steps=pasos, x=0.02, y=0.10, len=0.45,
                          currentvalue=dict(prefix="t = "))])
    else:
        for tr in estaticas:
            fig.add_trace(tr)

    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0),
                      scene=dict(aspectmode="data"),
                      legend=dict(x=0.01, y=0.99, font=dict(size=10)),
                      uirevision="camara")
    return fig


# ---------------------------------------------------------------- interfaz

app = Dash(__name__)
app.title = "T(x)=sRx+b, plano y boost O(3,1)"

def sldr(id_, val, mn, mx, step=0.1):
    return dcc.Slider(id=id_, min=mn, max=mx, step=step, value=val,
                      updatemode="drag", marks=None,
                      tooltip=dict(placement="bottom", always_visible=False))

def fila_vec(pref, etiqueta, vals, mn, mx):
    cols = []
    for i, ejer in enumerate("xyz"):
        cols.append(html.Div([html.Small(ejer, style={"color": "#666"}),
                              sldr(f"{pref}-{i}", vals[i], mn, mx)],
                             style={"flex": "1", "padding": "0 4px"}))
    return html.Div([html.Div(etiqueta, style={"fontSize": "12px",
                                               "fontWeight": "600"}),
                     html.Div(cols, style={"display": "flex"})],
                    style={"marginBottom": "2px"})

def bloque_par(pref, titulo, C, r, P, d, color):
    return html.Div([
        html.Div(titulo, style={"color": color, "fontWeight": "700",
                                "marginBottom": "4px"}),
        fila_vec(f"{pref}c", "Centro C", C, -8, 8),
        html.Div([html.Div("radio r", style={"fontSize": "12px",
                                             "fontWeight": "600"}),
                  sldr(f"{pref}-r", r, 0.4, 5, 0.05)],
                 style={"marginBottom": "2px"}),
        fila_vec(f"{pref}p", "Punto P de la recta", P, -8, 8),
        fila_vec(f"{pref}d", "Direccion d de la recta", d, -3, 3),
    ], style={"padding": "8px", "border": f"2px solid {color}",
              "borderRadius": "8px", "marginBottom": "8px"})

app.layout = html.Div([
    html.H3("T(x) = (sR)x + b  |  recta transformada + plano  |  boost de "
            "Lorentz O(3,1)", style={"marginBottom": "2px"}),
    html.P("Arrastra los sliders: todo se recalcula en tiempo real. T manda "
           "esfera->esfera exactamente, pero L1->T(L1) (naranja), que en general "
           "NO es L2. El plano gris por T(L1) y L2 pasa por C2. Marca la casilla "
           "para ver la animacion del morfismo.", style={"fontSize": "13px"}),
    html.Div([
        html.Div([
            bloque_par("p1", "Par 1 (origen: humero 1)",
                       [0, 0, 0], 1.0, [0.4, 0, -2], [0, 0, 1], AZUL),
            bloque_par("p2", "Par 2 (destino: humero 2)",
                       [5, 3, 1], 2.0, [5.5, 1, 1], [0, 1, 0.3], VERDE),
            dcc.Checklist(id="anim",
                          options=[{"label": " Ver animacion del morfismo "
                                             "(mas pesado; desmarcala para "
                                             "editar fluido)", "value": "on"}],
                          value=[], style={"fontSize": "12px", "margin": "4px 0"}),
            dcc.Loading(html.Div(id="panel"), type="dot"),
        ], style={"flex": "0 0 400px", "paddingRight": "12px",
                  "maxHeight": "95vh", "overflowY": "auto"}),
        dcc.Graph(id="g", style={"flex": "1", "height": "95vh"}),
    ], style={"display": "flex"}),
])

IDS = [f"p1c-{i}" for i in range(3)] + ["p1-r"] + \
      [f"p1p-{i}" for i in range(3)] + [f"p1d-{i}" for i in range(3)] + \
      [f"p2c-{i}" for i in range(3)] + ["p2-r"] + \
      [f"p2p-{i}" for i in range(3)] + [f"p2d-{i}" for i in range(3)]


def mat(R):
    if R is None:
        return "—"
    return html.Pre("\n".join("  ".join(f"{v: .3f}" for v in f) for f in R),
                    style={"margin": "2px 0", "fontSize": "12px",
                           "fontFamily": "monospace"})


def sec(titulo, hijos, ok=True):
    return html.Div([html.Div(titulo, style={"fontWeight": "700",
                                             "marginTop": "6px",
                                             "color": "#111" if ok else "#d62728"}),
                     *hijos],
                    style={"borderTop": "1px solid #ddd", "paddingTop": "4px"})


@app.callback(Output("g", "figure"), Output("panel", "children"),
              Input("anim", "value"), *[Input(i, "value") for i in IDS])
def actualizar(anim, *vals):
    if any(v is None for v in vals):
        return go.Figure(), "Completa los valores."
    (a0, a1, a2, r1, b0, b1_, b2, c0, c1, c2,
     d0, d1, d2, r2, e0, e1_, e2, f0, f1, f2) = vals
    par1 = analizar_par([a0, a1, a2], r1, [b0, b1_, b2], [c0, c1, c2])
    par2 = analizar_par([d0, d1, d2], r2, [e0, e1_, e2], [f0, f1, f2])
    trans = construir_T(par1, par2)
    linT = linea_transformada(par1, trans)
    plano = plano_por_lineas(linT, par2, trans) if linT else None
    boost = boost_lorentz(par1, par2)

    fig = figura(par1, par2, trans, linT, plano, boost,
                 con_anim=("on" in anim))

    okT = trans["valido"]
    rojo = {"color": "#d62728", "fontWeight": "700"}
    verde = {"color": "#2ca02c", "fontWeight": "700"}

    # ---- afín
    s_T = sec("1) Transformacion afin T(x)=sRx+b", [
        html.Div([html.B("s = r2/r1 = "),
                  html.Span(f"{trans['s']:.4f}",
                            style=({"fontSize": "18px"} if okT else
                                   {"fontSize": "18px", **rojo})),
                  html.Span("  posible" if okT else "  imposible",
                            style=(verde if okT else rojo))]),
        html.Div([html.B("R = "), mat(trans["R"])]),
        html.Div([html.B("b = "),
                  "—" if trans["b"] is None else
                  "(" + ", ".join(f"{v:.3f}" for v in trans["b"]) + ")"]),
        html.Div("R ortogonal: " +
                 ("—" if trans["err_orto"] is None else
                  f"||RR^T-I||={trans['err_orto']:.1e}, det={trans['det']:.4f} "
                  + ("ok" if trans["ortogonal"] else "NO")),
                 style={"fontSize": "12px"}),
        html.Ul([html.Li(p, style=rojo) for p in trans["problemas"]])
        if trans["problemas"] else html.Div(),
    ], ok=okT)

    # ---- recta transformada + plano
    if linT is not None and plano is not None:
        okL2 = plano["T_es_L2"]
        s_plano = sec("2) Recta transformada T(L1) y plano", [
            html.Div("T manda esfera1->esfera2 exactamente, pero L1->T(L1).",
                     style={"fontSize": "12px"}),
            html.Div([html.B("T(L1)=L2? "),
                      html.Span("SI (beta1=beta2)" if okL2 else "NO, en general",
                                style=(verde if okL2 else {"color": "#444"})),
                      html.Span(f"   [dist1={par1['dist']:.3f}, "
                               f"s*dist1={trans['s']*par1['dist']:.3f}, "
                               f"dist2={par2['dist']:.3f}]",
                               style={"fontSize": "11px", "color": "#666"})]),
            html.Div([html.B("Plano de T(L1) y L2 pasa por C2: "),
                      html.Span(f"residuo(C2)={plano['res_C2']:.1e}, "
                               f"residuo(L2)={plano['res_L2']:.1e} ok",
                               style=verde)], style={"fontSize": "12px"}),
        ])
    else:
        s_plano = sec("2) Recta transformada T(L1) y plano",
                      [html.Div("T no es valida (ver 1).", style=rojo)], ok=False)

    # ---- boost O(3,1)
    okB = boost["valido"]
    if okB:
        s_boost = sec("3) Boost de Lorentz  L en O(3,1)  (desde bases de Lorentz)", [
            html.Div([html.B("beta1="), f"{boost['b1']['beta']:.3f}  ",
                      html.B("beta2="), f"{boost['b2']['beta']:.3f}   "
                      f"(subluminal <=> la recta corta su esfera)"],
                     style={"fontSize": "12px"}),
            html.Div([html.B("Bases Lorentz ortonormales: "),
                      f"||F1^T eta F1 - eta||={boost['err_F1']:.1e}, "
                      f"||F2^T eta F2 - eta||={boost['err_F2']:.1e} ok"],
                     style={"fontSize": "12px"}),
            html.Div([html.B("Lambda = "), mat(boost["L"])]),
            html.Div([html.B("gamma_rel="), f"{boost['gamma_rel']:.4f}  ",
                      html.B("rapidez phi="), f"{boost['rapidez']:.4f}  ",
                      html.B("beta_rel="), f"{boost['beta_rel']:.4f}"],
                     style={"fontSize": "12px"}),
            html.Div([html.B("en O(3,1): "),
                      f"||L^T eta L - eta||={boost['err_O31']:.1e}, "
                      f"det={boost['det']:.4f}, Lambda00={boost['L00']:.3f}>=1 ",
                      html.Span("ok", style=verde)], style={"fontSize": "12px"}),
            html.Div([html.B("Lleva U1->U2: "),
                      f"||Lambda U1 - U2||={boost['mapea_err']:.1e} ",
                      html.Span("ok", style=verde)], style={"fontSize": "12px"}),
            html.Div([html.B("Boost puro (autovalores {e^-phi,1,1,e^+phi}): "),
                      "{" + ", ".join(f"{x:.3f}" for x in boost["eig"]) + "} ",
                      html.Span("ok" if boost["es_boost"] else "NO",
                                style=(verde if boost["es_boost"] else rojo))],
                     style={"fontSize": "12px"}),
        ])
    else:
        s_boost = sec("3) Boost de Lorentz L en O(3,1)",
                      [html.Ul([html.Li(p, style=rojo)
                                for p in boost["problemas"]])], ok=False)

    panel = html.Div([s_T, s_plano, s_boost],
                     style={"border": "1px solid #ccc", "borderRadius": "8px",
                            "padding": "10px", "background": "#fafafa",
                            "fontSize": "13px"})
    return fig, panel


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)
