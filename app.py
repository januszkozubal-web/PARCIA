# ATLAS — wariant Streamlit
# Parcia czynne na ściankę szczelną (logika z app.R)
# Autor: Janusz Witalis Kozubal
# Licencja: MIT License

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon as MplPolygon


def obliczenia(load: float, h: float, fi: float, gamma: float, coh: float) -> dict:
    ka = np.tan((45 - fi / 2) * np.pi / 180) ** 2
    # Odpór: dla tego samego modelu Rankine’a K_p = 1/K_a (φ takie samo)
    kp = 1.0 / max(ka, 1e-15)
    sqrt_kp = np.sqrt(kp)
    # Parcie bierne (pasywne): σ_p = K_p · σ_z + 2 c √K_p; σ_z z q_k, γ charakt.
    sigma_z_korona = load
    sigma_z_podstawa = load + gamma * h
    p_a = kp * sigma_z_podstawa + 2 * coh * sqrt_kp
    p_b = kp * sigma_z_korona + 2 * coh * sqrt_kp
    fh_p = 0.5 * (p_a + p_b) * h
    denom = p_a + p_b
    if denom != 0 and fh_p != 0:
        hyy_p = h * (p_a + 2 * p_b) / (3 * denom)
    else:
        hyy_p = float("nan")

    # Parcia czynne — wartości charakterystyczne: q, γ, c, φ (bez współczynników częściowych)
    q_k = load
    e_a = q_k * ka + ka * gamma * h - 2 * coh * np.sqrt(ka)
    e_b = q_k * ka - 2 * coh * np.sqrt(ka)

    if e_a > 0 and e_b > 0:
        m0 = h
        e_d = e_b
    elif e_b < 0 and e_a < 0:
        m0 = 0.0
        e_d = 0.0
    elif e_b < 0 and e_a > 0:
        m0 = h * abs(e_a) / (e_a + abs(e_b))
        e_d = 0.0
    else:
        # W oryginalnym R brak gałęzi dla części kombinacji — bezpieczny fallback
        m0 = h
        e_d = e_b

    fh = 0.5 * (e_a + e_d) * m0
    if fh != 0:
        hyy = (0.5 * m0**2 * e_d + (1 / 6) * m0**2 * (e_a - e_d)) / fh
    else:
        hyy = float("nan")

    return {
        "eA": e_a,
        "eB": e_b,
        "Hyy": hyy,
        "M0": m0,
        "eD": e_d,
        "Ka": ka,
        "Fh": fh,
        "H": h,
        "load": load,
        "Kp": kp,
        "pA": p_a,
        "pB": p_b,
        "Fh_passive": fh_p,
        "Hyy_passive": hyy_p,
    }


def draw_plot(w: dict, tryb: str) -> plt.Figure:
    """tryb: 'czynne' | 'pasywne' — wykres i opisy dopasowane do parć czynnych lub odporu."""
    h = w["H"]
    load = w["load"]
    fig, axp = plt.subplots(figsize=(8, 8))

    # Prostokąty jak w R (szerokość 5 m)
    df_figure_x = np.array([0, 5, 5, 0, 0])
    df_figure_y = np.array([0, 0, h, h, 0])
    axp.add_patch(
        MplPolygon(
            np.column_stack([df_figure_x, df_figure_y]),
            closed=True,
            facecolor="yellow",
            edgecolor="black",
            linewidth=0.5,
        )
    )

    df_rhomb_x = np.array([0, 5, 5, 0, 0])
    df_rhomb_y = np.array([h, h, h + load / 10, h + load / 10, h])
    axp.add_patch(
        MplPolygon(
            np.column_stack([df_rhomb_x, df_rhomb_y]),
            closed=True,
            facecolor="red",
            edgecolor="darkred",
            linewidth=0.5,
            hatch=r"|||",
        )
    )

    e_a, e_b = w["eA"], w["eB"]
    m0, e_d = w["M0"], w["eD"]
    p_a, p_b = w["pA"], w["pB"]

    if tryb == "czynne":
        df_pressure_an_x = np.array([0, e_a / 10, e_d / 10, 0, 0])
        df_pressure_an_y = np.array([0, 0, m0, m0, 0])
        axp.add_patch(
            MplPolygon(
                np.column_stack([df_pressure_an_x, df_pressure_an_y]),
                closed=True,
                facecolor="pink",
                edgecolor="deeppink",
                linewidth=0.5,
                hatch="---",
            )
        )

        axp.plot([0, 0], [0, h], color="black", linewidth=3, solid_capstyle="round")

        df_pressure_x = np.array([0, e_a / 10, e_b / 10, 0, 0])
        df_pressure_y = np.array([0, 0, h, h, 0])
        axp.add_patch(
            MplPolygon(
                np.column_stack([df_pressure_x, df_pressure_y]),
                closed=True,
                facecolor="gray",
                edgecolor="gray",
                alpha=0.5,
            )
        )

        for x0, y0 in [(e_a / 10, 0), (e_b / 10, h)]:
            axp.add_patch(
                FancyArrowPatch(
                    (x0, y0),
                    (0, y0),
                    arrowstyle="->",
                    mutation_scale=12,
                    color="blue",
                    linewidth=1,
                )
            )

        hyy, fh = w["Hyy"], w["Fh"]
        if not np.isnan(hyy) and fh != 0:
            axp.add_patch(
                FancyArrowPatch(
                    (fh / 30, hyy),
                    (0, hyy),
                    arrowstyle="->",
                    mutation_scale=14,
                    color="green",
                    linewidth=2,
                )
            )

        axp.plot([0], [m0], marker="o", color="purple", markersize=8, linestyle="none")
        tytul = "Ścianka szczelna — parcia czynne"
        opis_trybu = "PARCIE CZYNNE (Rankine, K_a)"
    else:
        # Pasywne: trapez σ od podstawy (pA) do korony (pB), pełna wysokość H (jak czynne: szary → ściana → różowy)
        df_pass_x = np.array([0, p_a / 10, p_b / 10, 0, 0])
        df_pass_y = np.array([0, 0, h, h, 0])
        axp.add_patch(
            MplPolygon(
                np.column_stack([df_pass_x, df_pass_y]),
                closed=True,
                facecolor="gray",
                edgecolor="gray",
                alpha=0.5,
            )
        )

        axp.plot([0, 0], [0, h], color="black", linewidth=3, solid_capstyle="round")

        axp.add_patch(
            MplPolygon(
                np.column_stack([df_pass_x, df_pass_y]),
                closed=True,
                facecolor="pink",
                edgecolor="deeppink",
                linewidth=0.5,
                hatch="---",
            )
        )

        for x0, y0 in [(p_a / 10, 0), (p_b / 10, h)]:
            axp.add_patch(
                FancyArrowPatch(
                    (x0, y0),
                    (0, y0),
                    arrowstyle="->",
                    mutation_scale=12,
                    color="blue",
                    linewidth=1,
                )
            )

        hyy_p, fh_p = w["Hyy_passive"], w["Fh_passive"]
        if not np.isnan(hyy_p) and fh_p != 0:
            axp.add_patch(
                FancyArrowPatch(
                    (fh_p / 30, hyy_p),
                    (0, hyy_p),
                    arrowstyle="->",
                    mutation_scale=14,
                    color="green",
                    linewidth=2,
                )
            )

        if not np.isnan(hyy_p):
            axp.plot([0], [hyy_p], marker="o", color="purple", markersize=8, linestyle="none")
        tytul = "Ścianka szczelna — odpór pasywny"
        opis_trybu = "ODPÓR PASYWNY (Rankine, K_p = 1/K_a)"

    axp.annotate("JVK 2024 MIT Licence", xy=(2, -0.2), fontsize=9, ha="center", va="top")
    axp.text(
        0.02,
        0.98,
        opis_trybu,
        transform=axp.transAxes,
        fontsize=11,
        fontweight="bold",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="black", alpha=0.92),
    )

    axp.set_aspect("equal", adjustable="box")
    axp.set_title(tytul)
    axp.set_xlabel("X")
    axp.set_ylabel("Y")
    axp.set_facecolor("white")
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    return fig


@st.dialog("Wyniki i założenia do obliczeń", width="large")
def wyniki_dialog(w: dict) -> None:
    hyy_txt = f"{w['Hyy']:.2f}" if not np.isnan(w["Hyy"]) else "—"
    st.markdown(
        f"""
Do obliczeń zastosowano założenia: ścianka gładka, naziom poziomy, obciążenie zmienne naziomu o stałej intensywności, skarpa pionowa, przemieszczenie nastepuje od gruntu.

**Wszystkie wielkości są liczone na wartościach charakterystycznych** (q, γ, c, φ — bez współczynników częściowych). Przy projektowaniu wg EN 1997 typowo stosuje się m.in. γ_Q = 1,5 na obciążenie zmienne i γ_φ = 1,35 na parametry gruntu — **nie są one wprowadzane w tej aplikacji**.

Analizę przeprowadzono dla 1 mb ściany. Kolor różowy parcia na ścianę, kolor szary parcia teoretyczne. Kolor żółty - grunt, kolor czerwony to obciżęnie skarpy. Wynikiem obliczeń jest wypadkowa - zielony wektor.

- Naprężenia teoretyczne na poziomie podstawy (kPa): **{w['eA']:.2f}**
- Naprężenia teoretyczne na wysokości korony skarpy (kPa): **{w['eB']:.2f}**
- Wynikająca siła pozioma F (kN): **{w['Fh']:.2f}** kN
- Poziom podstawy (m): **0.0 m**
- Wysokość samostatecznej części ściany mierząc od korony (m): **{w['H'] - w['M0']:.2f}**
- Wysokość działania siły poziomej F od podstawy (m): **{hyy_txt}**
- Wysokość skarpy H (m): **{w['H']:.2f}**
- Współczynnik rozdziału naprężeń czynnych K_a ( ): **{w['Ka']:.3f}**
- Obciążenie charakterystyczne korony q (kPa): **{w['load']:.2f}**
        """
    )


def main():
    st.set_page_config(page_title="ATLAS — parcia czynne", layout="wide")
    st.markdown(
        "[Kup mi kawę - utrzymanie serwera i wsparcie dla nowych programów](https://buycoffee.to/jan.vit)"
    )
    st.title("Parcia czynne dla skarpy z obciążeniem")

    col_side, col_main = st.columns([1, 2])

    with col_side:
        tryb = st.radio(
            "Wykres — stan gruntu:",
            ["czynne", "pasywne"],
            format_func=lambda x: "Parcia czynne (K_a)" if x == "czynne" else "Odpór pasywny (K_p = 1/K_a)",
            horizontal=True,
        )
        load = st.slider("Obciążenie korony q — charakt. (kPa):", 0.0, 20.0, 10.0, 1.0)
        h = st.slider("Wysokość skarpy H (m):", 1.0, 10.0, 3.0, 0.5)
        fi = st.slider("Kąt tarcia wewnętrznego gruntu fi (stopni):", 0, 40, 25, 1)
        gamma = st.slider("Ciężar obj. gruntu - gamma (kN/m^3):", 8.0, 25.0, 20.0, 0.5)
        coh = st.slider("Spójność (kPa):", 0.0, 50.0, 5.0, 1.0)
        show_clicked = st.button("Pokaż wyniki i założenia")

    wyniki = obliczenia(load, h, fi, gamma, coh)

    with col_side:
        with st.expander("Odpór (K_p = 1/K_a)", expanded=False):
            st.caption(
                "Wartości charakterystyczne. σ_p = K_p·σ_z + 2c√K_p; σ_z = q+γH u podstawy, σ_z = q u korony; K_p = 1/K_a."
            )
            st.metric("K_p", f"{wyniki['Kp']:.4f}")
            c1, c2 = st.columns(2)
            c1.metric("σ_p u podstawy (kPa)", f"{wyniki['pA']:.2f}")
            c2.metric("σ_p u korony (kPa)", f"{wyniki['pB']:.2f}")
            c3, c4 = st.columns(2)
            hy_p_txt = (
                f"{wyniki['Hyy_passive']:.2f}"
                if not np.isnan(wyniki["Hyy_passive"])
                else "—"
            )
            c3.metric("Siła odporu R (kN/m)", f"{wyniki['Fh_passive']:.2f}")
            c4.metric("Wys. R od podstawy (m)", hy_p_txt)

    with col_main:
        fig = draw_plot(wyniki, tryb)
        st.pyplot(fig)
        plt.close(fig)

    if show_clicked:
        wyniki_dialog(wyniki)

    st.caption(
        "ATLAS — Streamlit; obliczenia na wartościach charakterystycznych (bez γ=1,5/1,35 w wzorach). "
        "Oryginalna aplikacja Shiny stosowała q·(1,5/1,35) przy parciu czynnym."
    )


if __name__ == "__main__":
    main()
