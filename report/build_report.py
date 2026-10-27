"""Builds report/report.html from results/*.csv (inline SVG charts + MathJax equations)."""
import html
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
LT = ["Conventional", "FHA", "VA", "USDA"]
LTC = {"Conventional": "c1", "FHA": "c2", "VA": "c3", "USDA": "c4"}
MARK = {"Conventional": "circle", "FHA": "square", "VA": "diamond", "USDA": "tri"}


def f(x, n=3):
    return f"{x:.{n}f}"


def pct(x, n=1):
    return f"{100 * x:.{n}f}%"


def esc(s):
    return html.escape(str(s))


def marker(kind, x, y, cls, hollow=False, r=4.2):
    fill = "var(--bg)" if hollow else "currentColor"
    a = f'class="{cls}" fill="{fill}" stroke="currentColor" stroke-width="1.6"'
    if kind == "circle":
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" {a}/>'
    if kind == "square":
        return f'<rect x="{x - r:.1f}" y="{y - r:.1f}" width="{2 * r}" height="{2 * r}" {a}/>'
    if kind == "diamond":
        return f'<path d="M{x:.1f} {y - r - 1:.1f}L{x + r + 1:.1f} {y:.1f}L{x:.1f} {y + r + 1:.1f}L{x - r - 1:.1f} {y:.1f}Z" {a}/>'
    return f'<path d="M{x:.1f} {y - r - 1:.1f}L{x + r + 1:.1f} {y + r:.1f}L{x - r - 1:.1f} {y + r:.1f}Z" {a}/>'


# ------------------------------------------------------------------ data
funnel = pd.read_csv(R / "funnel.csv")
met = pd.read_csv(R / "metrics.csv")
ov = met[met.subgroup == "overall"]
sk = ov[ov.engine == "sklearn"].set_index("model")
base_row = ov[ov.model == "const"].iloc[0]
odds = pd.read_csv(R / "odds_ratios_A.csv")
dti = pd.read_csv(R / "adjusted_rate_dti_by_program.csv")
cltv = pd.read_csv(R / "adjusted_rate_cltv_by_program.csv")
heat = pd.read_csv(R / "adjusted_rate_dti_x_cltv.csv")
cal = pd.read_csv(R / "calibration.csv")
eq = pd.read_csv(R / "equivalence.csv")
thr = pd.read_csv(R / "threshold_table.csv")
fis = pd.read_csv(R / "fit_info_sklearn.csv").set_index("tag")
fip = pd.read_csv(R / "fit_info_spark.csv").set_index("tag")
bylt = met[(met.engine == "sklearn") & met.model.isin(["A", "B", "C", "G"]) & (met.subgroup != "overall")]
prog = met[(met.engine == "sklearn") & met.model.str.startswith("B_")].set_index("model")
N = int(funnel.rows.iloc[-1])


# ------------------------------------------------------------------ charts
def svg_open(w, h, label):
    return f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{esc(label)}" class="chart">'


def chart_funnel():
    labels = ["All 2024 HMDA rows", "Decided applications (action 1, 2 or 3)",
              "Excl. reverse, open-end, business", "Home purchase, first lien, principal residence",
              "Site-built, 1–4 units (analysis set)"]
    w, h = 760, 250
    mx = funnel.rows.max()
    out = [svg_open(w, h, "Cohort funnel")]
    for i, (lab, n) in enumerate(zip(labels, funnel.rows)):
        y = 14 + i * 46
        bw = (w - 24) * n / mx
        out.append(f'<text x="12" y="{y + 2}" class="t-lab">{esc(lab)}</text>')
        out.append(f'<rect x="12" y="{y + 8}" width="{bw:.1f}" height="20" class="bar{" bar-hl" if i == 4 else ""}"/>')
        txt = f"{int(n):,}"
        if bw > 120:
            out.append(f'<text x="20" y="{y + 22}" class="t-num t-on">{txt}</text>')
        else:
            out.append(f'<text x="{12 + bw + 8:.1f}" y="{y + 22}" class="t-num">{txt}</text>')
    return "".join(out) + "</svg>"


MODELS_DOT = [("A", "A  main effects", "m"), ("B", "B  + DTI×CLTV", "m"), ("C", "C  + program slopes", "m"),
              ("D", "D  C, finer DTI", "m"), ("E", "E  C + state", "m"), ("F", "F  C, CLTV≤150", "m"),
              ("G", "G  boosting", "g")]


def chart_dots(metric, title, xmin, xmax, ticks, fmtn=3, base=None):
    w, rowh, top, left = 720, 32, 26, 160
    h = top + rowh * len(MODELS_DOT) + 34

    def X(v):
        return left + (w - left - 24) * (v - xmin) / (xmax - xmin)

    out = [svg_open(w, h, title)]
    for t in ticks:
        out.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{top - 8}" y2="{h - 30}" class="grid"/>')
        out.append(f'<text x="{X(t):.1f}" y="{h - 12}" class="t-ax" text-anchor="middle">{t:.2f}</text>')
    for i, (m, lab, cls) in enumerate(MODELS_DOT):
        y = top + i * rowh + rowh / 2
        r = sk.loc[m]
        v = r[metric]
        out.append(f'<text x="{left - 12}" y="{y + 4:.1f}" class="t-lab" text-anchor="end">{esc(lab)}</text>')
        lo, hi = r[metric + "_lo"], r[metric + "_hi"]
        out.append(f'<line x1="{X(lo):.1f}" x2="{X(hi):.1f}" y1="{y:.1f}" y2="{y:.1f}" class="ci {cls}"/>')
        out.append(f'<circle cx="{X(v):.1f}" cy="{y:.1f}" r="5" class="dot {cls}"/>')
        out.append(f'<text x="{X(v):.1f}" y="{y - 9:.1f}" class="t-num" text-anchor="middle">{v:.{fmtn}f}</text>')
    return "".join(out) + "</svg>"


def chart_forest():
    d = odds[odds.feature != "intercept"].copy()
    names = {"loan_type=FHA": "FHA (vs Conventional)", "loan_type=VA": "VA", "loan_type=USDA": "USDA",
             "dti_bin=36-42": "DTI 36–42 (vs <36)", "dti_bin=43-49": "DTI 43–49", "dti_bin=50-60": "DTI 50–60",
             "dti_bin=>60": "DTI >60", "dti_bin=Missing": "DTI missing",
             "cltv_bin=80-90": "CLTV 80–90 (vs ≤80)", "cltv_bin=90-95": "CLTV 90–95",
             "cltv_bin=95-100": "CLTV 95–100", "cltv_bin=>100": "CLTV >100", "cltv_bin=Missing": "CLTV missing",
             "log_income": "log income (per SD)", "income_missing": "income missing",
             "log_loan_amount": "log loan amount (per SD)"}
    w, rowh, top, left = 720, 24, 14, 190
    h = top + rowh * len(d) + 44
    lo, hi = math.log(0.3), math.log(160)

    def X(v):
        return left + (w - left - 40) * (math.log(v) - lo) / (hi - lo)

    out = [svg_open(w, h, "Odds ratios, Model A")]
    for t in [0.5, 1, 2, 5, 10, 25, 100]:
        out.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{top - 6}" y2="{h - 34}" class="{"ref" if t == 1 else "grid"}"/>')
        out.append(f'<text x="{X(t):.1f}" y="{h - 18}" class="t-ax" text-anchor="middle">{t:g}</text>')
    out.append(f'<text x="{(left + w - 40) / 2:.1f}" y="{h - 3}" class="t-ax" text-anchor="middle">odds ratio of denial (log scale)</text>')
    for i, r in enumerate(d.itertuples()):
        y = top + i * rowh + rowh / 2
        grp = "c1" if r.feature.startswith("dti") else ("c2" if r.feature.startswith("cltv") else ("c3" if r.feature.startswith("loan_type") else "c4"))
        out.append(f'<text x="{left - 10}" y="{y + 4:.1f}" class="t-lab" text-anchor="end">{esc(names[r.feature])}</text>')
        out.append(f'<line x1="{X(r.or_ci_low):.1f}" x2="{X(r.or_ci_high):.1f}" y1="{y:.1f}" y2="{y:.1f}" class="ci {grp}"/>')
        out.append(f'<circle cx="{X(r.odds_ratio):.1f}" cy="{y:.1f}" r="4.2" class="dot {grp}"/>')
        out.append(f'<text x="{X(r.or_ci_high) + 7:.1f}" y="{y + 4:.1f}" class="t-num">{r.odds_ratio:.2f}</text>')
    return "".join(out) + "</svg>"


def chart_lines(tbl, col, order, xlabel, ymax=1.0, step=0.2):
    w, h, L, Rm, T, B = 720, 330, 54, 20, 22, 54

    def X(i):
        return L + (w - L - Rm) * (i + 0.5) / len(order)

    def Y(v):
        return T + (h - T - B) * (1 - v / ymax)

    out = [svg_open(w, h, "Adjusted denial rate by " + xlabel)]
    for t in [round(k * step, 4) for k in range(int(round(ymax / step)) + 1)]:
        out.append(f'<line x1="{L}" x2="{w - Rm}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" class="grid"/>')
        out.append(f'<text x="{L - 8}" y="{Y(t) + 4:.1f}" class="t-ax" text-anchor="end">{t * 100:g}%</text>')
    for i, b in enumerate(order):
        out.append(f'<text x="{X(i):.1f}" y="{h - B + 20}" class="t-ax" text-anchor="middle">{esc(b.replace("<=", "≤"))}</text>')
    out.append(f'<text x="{(L + w - Rm) / 2:.1f}" y="{h - 8}" class="t-ax" text-anchor="middle">{esc(xlabel)}</text>')
    ncol = [c for c in tbl.columns if c.startswith("observed_n")][0]
    for lt in LT:
        g = tbl[tbl.loan_type == lt].set_index(col).reindex(order)
        pts = [(X(i), Y(v)) for i, v in enumerate(g.adjusted_denial_rate)]
        out.append(f'<g class="{LTC[lt]}"><polyline points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in pts)}" fill="none" stroke="currentColor" stroke-width="2.2"/>')
        for (a, b), n in zip(pts, g[ncol]):
            out.append(marker(MARK[lt], a, b, LTC[lt], hollow=n < 500))
        out.append("</g>")
    return "".join(out) + "</svg>"


def legend():
    items = "".join(
        f'<span class="lg {LTC[lt]}"><svg width="18" height="14" viewBox="0 0 18 14">{marker(MARK[lt], 9, 7, LTC[lt])}</svg>{lt}</span>'
        for lt in LT)
    return f'<div class="legend">{items}<span class="lg mut"><svg width="18" height="14" viewBox="0 0 18 14">{marker("circle", 9, 7, "mut", hollow=True)}</svg>hollow = under 500 rows</span></div>'


DTI_AX = ["<36", "36-42", "43-49", "50-60", ">60"]
CLTV_AX = ["<=80", "80-90", "90-95", "95-100", ">100"]
DB = ["<36", "36-42", "43-49", "50-60", ">60", "Missing"]
CB = ["<=80", "80-90", "90-95", "95-100", ">100", "Missing"]


def heatmap(panel, title):
    d = heat[heat.panel == panel].set_index(["dti_bin", "cltv_bin"])
    cs, L, T = 40, 62, 30
    w, h = L + cs * 6 + 8, T + cs * 6 + 44
    out = [svg_open(w, h, "Heatmap " + title)]
    out.append(f'<text x="{L}" y="14" class="t-lab" font-weight="600">{esc(title)}</text>')
    for j, c in enumerate(CB):
        out.append(f'<text x="{L + cs * j + cs / 2}" y="{T + cs * 6 + 16}" class="t-ax" text-anchor="middle">{esc(c.replace("<=", "≤").replace("Missing", "Miss."))}</text>')
    out.append(f'<text x="{L + cs * 3}" y="{h - 4}" class="t-ax" text-anchor="middle">CLTV %</text>')
    for i, dd in enumerate(DB):
        out.append(f'<text x="{L - 6}" y="{T + cs * i + cs / 2 + 4}" class="t-ax" text-anchor="end">{esc(dd.replace("Missing", "Miss."))}</text>')
        for j, c in enumerate(CB):
            r = d.loc[(dd, c)]
            x, y = L + cs * j, T + cs * i
            if r.observed_n < 500:
                out.append(f'<rect x="{x + 1}" y="{y + 1}" width="{cs - 2}" height="{cs - 2}" class="mask"/>')
                continue
            v = r.adjusted_denial_rate
            out.append(f'<rect x="{x + 1}" y="{y + 1}" width="{cs - 2}" height="{cs - 2}" class="heat" fill-opacity="{0.06 + 0.94 * v:.3f}"/>')
            out.append(f'<text x="{x + cs / 2}" y="{y + cs / 2 + 4}" class="t-cell{" hi" if v > 0.5 else ""}" text-anchor="middle">{v * 100:.0f}</text>')
    out.append(f'<text transform="translate(10 {T + cs * 3}) rotate(-90)" class="t-ax" text-anchor="middle">DTI %</text>')
    return "".join(out) + "</svg>"


def chart_cal():
    w, h, L, B, T = 560, 460, 56, 44, 14
    lo, hi = math.log(0.015), math.log(0.6)

    def X(v):
        return L + (w - L - 14) * (math.log(v) - lo) / (hi - lo)

    def Y(v):
        return T + (h - T - B) * (1 - (math.log(v) - lo) / (hi - lo))

    out = [svg_open(w, h, "Calibration by decile, log-log axes")]
    for t in [0.02, 0.05, 0.1, 0.2, 0.4]:
        out.append(f'<line x1="{L}" x2="{w - 14}" y1="{Y(t):.1f}" y2="{Y(t):.1f}" class="grid"/>')
        out.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{T}" y2="{h - B}" class="grid"/>')
        out.append(f'<text x="{L - 8}" y="{Y(t) + 4:.1f}" class="t-ax" text-anchor="end">{t * 100:g}%</text>')
        out.append(f'<text x="{X(t):.1f}" y="{h - B + 18}" class="t-ax" text-anchor="middle">{t * 100:g}%</text>')
    out.append(f'<line x1="{X(0.015):.1f}" y1="{Y(0.015):.1f}" x2="{X(0.6):.1f}" y2="{Y(0.6):.1f}" class="ref"/>')
    out.append(f'<text x="{(L + w) / 2}" y="{h - 6}" class="t-ax" text-anchor="middle">mean predicted denial rate (test-set decile, log scale)</text>')
    out.append(f'<text transform="translate(14 {(T + h - B) / 2}) rotate(-90)" class="t-ax" text-anchor="middle">observed denial rate (log scale)</text>')
    for m, cls, mk in [("A", "c1", "circle"), ("C", "c3", "square"), ("G", "c2", "diamond")]:
        g = cal[(cal.model == m) & (cal.engine == "sklearn")].sort_values("decile")
        pts = [(X(a), Y(b)) for a, b in zip(g.mean_predicted, g.observed_rate)]
        out.append(f'<g class="{cls}"><polyline points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in pts)}" fill="none" stroke="currentColor" stroke-width="1.4" opacity=".55"/>')
        for a, b in pts:
            out.append(marker(mk, a, b, cls, r=3.4))
        out.append("</g>")
    return "".join(out) + "</svg>"


# ------------------------------------------------------------------ tables
def metrics_table():
    rows = [f'<tr><td>Constant (train rate)</td><td class="n">—</td><td class="n">0.500</td><td class="n">{f(base_row.pr_auc)}</td>'
            f'<td class="n">{f(base_row.log_loss, 4)}</td><td class="n">{f(base_row.brier, 4)}</td></tr>']
    for m, lab, _ in MODELS_DOT:
        r = sk.loc[m]
        cols = f'{int(fis.loc[m, "n_cols"])}' if m != "G" else "—"
        rows.append(f'<tr><td><b>{m}</b> <span class="mut">{esc(lab[3:])}</span></td><td class="n">{cols}</td>'
                    f'<td class="n">{f(r.roc_auc)} <span class="mut">[{f(r.roc_auc_lo)}, {f(r.roc_auc_hi)}]</span></td>'
                    f'<td class="n">{f(r.pr_auc)} <span class="mut">[{f(r.pr_auc_lo)}, {f(r.pr_auc_hi)}]</span></td>'
                    f'<td class="n">{f(r.log_loss, 4)}</td><td class="n">{f(r.brier, 4)}</td></tr>')
    return ('<div class="scroll"><table><thead><tr><th>Model</th><th class="n">Columns</th><th class="n">ROC-AUC [95% CI]</th>'
            '<th class="n">PR-AUC [95% CI]</th><th class="n">Log loss</th><th class="n">Brier</th></tr></thead><tbody>'
            + "".join(rows) + "</tbody></table></div>")


def lt_table():
    rows = []
    for lt in LT:
        cells = []
        for m in ["A", "C", "G"]:
            r = bylt[(bylt.model == m) & (bylt.subgroup == lt)].iloc[0]
            cells.append(f'<td class="n">{f(r.roc_auc)}</td><td class="n">{f(r.pr_auc)}</td>')
        r0 = bylt[(bylt.model == "A") & (bylt.subgroup == lt)].iloc[0]
        rows.append(f'<tr><td>{lt}</td><td class="n">{int(r0.n):,}</td><td class="n">{pct(r0.denial_rate)}</td>' + "".join(cells) + "</tr>")
    return ('<div class="scroll"><table><thead><tr><th rowspan="2">Program</th><th class="n" rowspan="2">Test rows</th><th class="n" rowspan="2">Denied</th>'
            '<th class="n" colspan="2">Model A</th><th class="n" colspan="2">Model C</th><th class="n" colspan="2">Model G</th></tr>'
            '<tr><th class="n">ROC</th><th class="n">PR</th><th class="n">ROC</th><th class="n">PR</th><th class="n">ROC</th><th class="n">PR</th></tr></thead><tbody>'
            + "".join(rows) + "</tbody></table></div>")


def eq_table():
    rows = []
    for m in ["A", "B", "C"]:
        e = eq[eq.model == m].set_index("quantity")
        cf, ic, ra = e.loc["max abs coefficient difference"], e.loc["intercept difference"], e.loc["test ROC-AUC"]
        ll, cr = e.loc["test log loss"], e.loc["Pearson corr of test scores"]

        def ok(r):
            return '<span class="ok">pass</span>' if r["pass"] else '<span class="bad">fail</span>'

        rows.append(f'<tr><td><b>{m}</b></td><td class="n">{cf.difference:.1e}</td><td>{ok(cf)}</td><td class="n">{ic.difference:.1e}</td><td>{ok(ic)}</td>'
                    f'<td class="n">{ra.difference:.1e}</td><td>{ok(ra)}</td><td class="n">{ll.difference:.1e}</td><td>{ok(ll)}</td>'
                    f'<td class="n">{float(cr.sklearn):.8f}</td><td>{ok(cr)}</td><td class="n">{int(fis.loc[m, "n_iter"])} / {int(fip.loc[m, "n_iter"])}</td></tr>')
    return ('<div class="scroll"><table><thead><tr><th>Model</th><th class="n" colspan="2">max coef diff (≤1e-3)</th><th class="n" colspan="2">intercept diff (≤1e-3)</th>'
            '<th class="n" colspan="2">ROC-AUC diff (≤1e-4)</th><th class="n" colspan="2">log loss diff (≤1e-4)</th><th class="n" colspan="2">score correlation (≥0.9999)</th>'
            '<th class="n">iterations sklearn / Spark</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table></div>")


def prog_table():
    rows = []
    for lt in LT:
        r = prog.loc["B_" + lt]
        rows.append(f'<tr><td>{lt}</td><td class="n">{int(r.n):,}</td><td class="n">{f(r.roc_auc)} <span class="mut">[{f(r.roc_auc_lo)}, {f(r.roc_auc_hi)}]</span></td>'
                    f'<td class="n">{f(r.pr_auc)}</td><td class="n">{f(r.log_loss, 4)}</td></tr>')
    return ('<div class="scroll"><table><thead><tr><th>Model B fit within one program</th><th class="n">Test rows</th><th class="n">ROC-AUC [95% CI]</th>'
            '<th class="n">PR-AUC</th><th class="n">Log loss</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table></div>")


def feat_matrix():
    cols = list("ABCDEFG")
    rowsdef = [
        ("Loan type, DTI bin, CLTV bin, log income, income missing, log loan amount", "XXXXXXX"),
        ("DTI×CLTV cross (rare cells pooled)", ".XXXXX."),
        ("Loan type × DTI, loan type × CLTV", "..XXXX."),
        ("Finer DTI (each integer 36–49)", "...X..."),
        ("State dummies", "....X.."),
        ("Drop rows with CLTV above 150", ".....X."),
    ]
    head = "".join(f"<th>{c}</th>" for c in cols)
    body = ""
    for lab, pat in rowsdef:
        tds = "".join('<td class="dotcell">●</td>' if ch == "X" else '<td class="dotcell off">·</td>' for ch in pat)
        body += f"<tr><td>{esc(lab)}</td>{tds}</tr>"
    body += "<tr><td>Learner</td>" + "".join(f'<td class="dotcell sm">{"GBM" if c == "G" else "LR"}</td>' for c in cols) + "</tr>"
    body += "<tr><td>Fit in</td>" + "".join(f'<td class="dotcell sm">{"sk" if c == "G" else "sk+Spark"}</td>' for c in cols) + "</tr>"
    return f'<div class="scroll"><table class="matrix"><thead><tr><th></th>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def thr_table():
    rows = ""
    for v in ["DTI", "CLTV"]:
        for lt in LT:
            r = thr[(thr.variable == v) & (thr.loan_type == lt)].iloc[0]
            b = r.first_bin_with_jump
            has = pd.notna(b)
            rows += (f'<tr><td>{v}</td><td>{lt}</td><td>{esc(b) if has else "none"}</td>'
                     f'<td class="n">{("+%.0f pp" % r.jump_pp) if has else chr(8212)}</td></tr>')
    return ('<div class="scroll"><table><thead><tr><th>Variable</th><th>Program</th><th>First bin with a rise above 5 pp</th><th class="n">Rise</th></tr></thead><tbody>'
            + rows + "</tbody></table></div>")


# ------------------------------------------------------------------ numbers used in text
A, B_, C_, D_, E_, F_, G_ = (sk.loc[m] for m in "ABCDEFG")
den = pd.read_parquet(ROOT / "data" / "analysis.parquet", columns=["denied", "loan_type"])
rate = den.denied.mean()
rate_lt = den.groupby(pd.to_numeric(den.loan_type)).denied.mean()


def dti_get(lt, b):
    return dti[(dti.loan_type == lt) & (dti.dti_bin == b)].iloc[0].adjusted_denial_rate


def cl_get(lt, b):
    return cltv[(cltv.loan_type == lt) & (cltv.cltv_bin == b)].iloc[0].adjusted_denial_rate


def orv(name):
    return odds[odds.feature == name].odds_ratio.iloc[0]


tmpl = (Path(__file__).parent / "template.html").read_text(encoding="utf-8")
rep = {
    "%%FUNNEL%%": chart_funnel(),
    "%%DOT_ROC%%": chart_dots("roc_auc", "ROC-AUC by model", 0.72, 0.82, [0.72, 0.75, 0.78, 0.81]),
    "%%DOT_PR%%": chart_dots("pr_auc", "PR-AUC by model", 0.40, 0.55, [0.40, 0.45, 0.50, 0.55]),
    "%%FOREST%%": chart_forest(),
    "%%LEGEND%%": legend(),
    "%%DTI_LINES%%": chart_lines(dti, "dti_bin", DTI_AX, "DTI bin (%)"),
    "%%CLTV_LINES%%": chart_lines(cltv, "cltv_bin", CLTV_AX, "CLTV bin (%)", ymax=0.30, step=0.05),
    "%%HEAT_ALL%%": heatmap("All programs", "All programs (Model B)"),
    "%%HEAT_C%%": heatmap("Conventional", "Conventional"), "%%HEAT_F%%": heatmap("FHA", "FHA"),
    "%%HEAT_V%%": heatmap("VA", "VA"), "%%HEAT_U%%": heatmap("USDA", "USDA"),
    "%%CAL%%": chart_cal(),
    "%%METRICS%%": metrics_table(), "%%LTTABLE%%": lt_table(), "%%EQ%%": eq_table(), "%%PROG%%": prog_table(),
    "%%MATRIX%%": feat_matrix(), "%%THR%%": thr_table(),
    "%%N%%": f"{N:,}", "%%NTRAIN%%": f"{int(fis.loc['A', 'n_train']):,}", "%%NTEST%%": f"{int(fis.loc['A', 'n_test']):,}",
    "%%RATE%%": pct(rate, 2),
    "%%R_C%%": pct(rate_lt[1], 2), "%%R_F%%": pct(rate_lt[2], 2), "%%R_V%%": pct(rate_lt[3], 2), "%%R_U%%": pct(rate_lt[4], 2),
    "%%A_ROC%%": f(A.roc_auc), "%%A_PR%%": f(A.pr_auc), "%%C_ROC%%": f(C_.roc_auc), "%%C_PR%%": f(C_.pr_auc),
    "%%G_ROC%%": f(G_.roc_auc), "%%G_PR%%": f(G_.pr_auc), "%%E_ROC%%": f(E_.roc_auc), "%%BASE_PR%%": f(base_row.pr_auc),
    "%%GAP%%": f(G_.pr_auc - C_.pr_auc),
    "%%DTI_CONV_LOW%%": pct(dti_get("Conventional", "43-49")), "%%DTI_CONV_HI%%": pct(dti_get("Conventional", "50-60"), 0),
    "%%DTI_FHA_HI%%": pct(dti_get("FHA", ">60"), 0), "%%DTI_FHA_MID%%": pct(dti_get("FHA", "50-60"), 0),
    "%%DTI_VA_HI%%": pct(dti_get("VA", ">60"), 0), "%%DTI_VA_MID%%": pct(dti_get("VA", "50-60"), 1),
    "%%DTI_USDA_43%%": pct(dti_get("USDA", "43-49"), 0), "%%DTI_USDA_36%%": pct(dti_get("USDA", "36-42"), 0),
    "%%CL_CONV_LO%%": pct(cl_get("Conventional", "<=80")), "%%CL_CONV_HI%%": pct(cl_get("Conventional", ">100")),
    "%%CL_VA_HI%%": pct(cl_get("VA", ">100"), 0), "%%CL_VA_LO%%": pct(cl_get("VA", "95-100")),
    "%%OR_DTI60%%": f"{orv('dti_bin=>60'):.0f}", "%%OR_DTI5060%%": f"{orv('dti_bin=50-60'):.1f}",
    "%%OR_VA%%": f"{orv('loan_type=VA'):.2f}", "%%OR_CLTV100%%": f"{orv('cltv_bin=>100'):.2f}",
    "%%OR_INCM%%": f"{orv('income_missing'):.1f}",
    "%%FIT_A_SK%%": f"{fis.loc['A', 'fit_seconds']:.0f}", "%%FIT_A_SP%%": f"{fip.loc['A', 'fit_seconds']:.0f}",
    "%%FIT_C_SK%%": f"{fis.loc['C', 'fit_seconds']:.0f}", "%%FIT_C_SP%%": f"{fip.loc['C', 'fit_seconds']:.0f}",
}
for k, v in rep.items():
    tmpl = tmpl.replace(k, v)
left = [t for t in tmpl.replace(">", "> ").split() if t.startswith("%%")]
assert not left, left
(Path(__file__).parent / "report.html").write_text(tmpl, encoding="utf-8")
print("ok", len(tmpl))
