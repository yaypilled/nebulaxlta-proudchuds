"""Presentation layer for the unified monitoring workflow."""
from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import streamlit as st
from src.app.validators import timestamp

ROOT = Path(__file__).resolve().parents[2]
INFO = {
    "door": {
        "title": "Door System", "tag": "CYCLE DETECTION",
        "summary": "Find door cycles and flag abnormal mechanical resistance.",
        "input": "One continuous Door CSV with timestamps and controller headers.",
        "result": "Cycle boundaries and Normal / Abnormal resistance labels",
        "method": "Gaps in the controller stream identify individual cycles. The sum of motor-current readings is compared with the fitted threshold for opening or closing. Greater cumulative current is a proxy for increased motor effort. Thresholds are fitted to the supplied training data.",
        "scope": "110 labelled training cycles form a small evidence base. Segmentation assumes gaps between cycles, as in the supplied stream format. Results guide inspection and do not confirm a physical defect.",
    },
    "acv": {
        "title": "ACV System", "tag": "FAULT LOCALISATION",
        "summary": "Rank all eight cars for refrigerant-leak inspection.",
        "input": "An Excel (.xlsx) telemetry file with all eight cars, using the standard temperature and control-mode schema.",
        "result": "A complete ranking from highest to lowest inspection priority",
        "method": "Each car's temperature and cooling-control behaviour is compared with its peers. Six fixed methods combine thermal comparisons, three fitted classifiers and persistence across 30- and 60-minute windows, with equal weights.",
        "scope": "Only five independent training cases share the standard schema. The richer case-04 schema is outside this model's scope. Suspicion scores are relative rankings, not calibrated failure probabilities.",
    },
    "rail": {
        "title": "Rail Corrugation", "tag": "SIDE CLASSIFICATION",
        "summary": "Detect vibration patterns associated with rail corrugation.",
        "input": "One or more Rail CSV files, each with a header and 10,000 readings across 129 columns.",
        "result": "Normal, Side I or Side II for each recording",
        "method": "The fitted pipeline compares vibration and shock from axle boxes on each rail side. Features account for speed before classification. Below 1.28 m/s, a documented low-speed rule returns Normal without invoking the classifier.",
        "scope": "A low-speed Normal result reflects the model's domain rule and does not confirm a defect-free rail. Processing errors block output generation. Test labels are held by the organisers.",
    },
    "shm": {
        "title": "Structural Health", "tag": "FATIGUE ESTIMATION",
        "summary": "Estimate cumulative fatigue damage from dynamic stress.",
        "input": "One or more SHM CSV files, each with one numeric stress column and no header.",
        "result": "One cumulative fatigue-damage estimate per file",
        "method": "Rainflow counting summarises stress cycles with 64 stress classes and repeat-history residue closure. A fitted ridge model combines fatigue features at several stress exponents to estimate cumulative damage.",
        "scope": "Training data cover healthy operating conditions. Estimates apply to the supplied recording segment. They do not establish remaining useful life, safety certification or a confirmed structural fault.",
    },
}
COLOURS = {"Normal": "#218477", "Abnormal resistance": "#D57729", "Side I": "#D57729", "Side II": "#356AA8"}


def chart(fig, height=340):
    fig.update_layout(height=height, margin=dict(l=12, r=20, t=18, b=15),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial, sans-serif", size=13, color="#23374D"),
        legend_title_text="", legend=dict(orientation="h", y=1.15, x=0))
    fig.update_xaxes(gridcolor="#E7EBF0", zeroline=False)
    fig.update_yaxes(gridcolor="#E7EBF0", zeroline=False)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def explain(key):
    with st.expander("How was this result produced?"):
        st.write(INFO[key]["method"])
    with st.expander("Validation and model scope"):
        if key == "door":
            st.write("Committed rolling-origin validation reports five folds at 1.000 IoU-weighted F1. "
                     "The always-Normal baseline with correct boundaries scores 0.7273. "
                     "These are local validation results, not held-out test scores.")
        elif key == "acv":
            evidence = pd.read_csv(ROOT / "artifacts/acv/acv_locked_validation.csv")
            st.write(f"Committed leave-one-case-out rank-decay: {evidence.rank_decay_score.mean():.3f} "
                     f"across {len(evidence)} comparable cases. This very small sample does not establish test accuracy.")
        elif key == "rail":
            st.write("Committed repeated cross-validation macro F1: 0.7765 on the restricted domain "
                     "and 0.7858 on the full eligible domain. These use different validation populations. "
                     "The evaluation code and evidence are preserved in the repository.")
        else:
            record = ROOT / "artifacts/shm/reconstruction.json"
            if record.exists():
                data = json.loads(record.read_text())["reconstruction"]
                st.write(f"Reconstructed from {data['training_files']} labelled training files. "
                         f"All {data['parity_files']} outputs reproduce the previously committed predictions. "
                         "This checks artifact reconstruction, not independent test performance.")
        st.write(INFO[key]["scope"])


def show_door(result):
    detail = next(iter(result.diagnostics["by_file"].values()))
    cycles = pd.DataFrame(detail["cycles"])
    abnormal = int(cycles.prediction.eq("Abnormal resistance").sum())
    for col, label, value in zip(st.columns(5),
            ["Cycles detected", "Normal", "Abnormal resistance", "Open", "Close"],
            [len(cycles), len(cycles)-abnormal, abnormal,
             int(cycles.operation.eq("Open").sum()), int(cycles.operation.eq("Close").sum())]):
        col.metric(label, value, border=True)
    if abnormal:
        st.warning(f"Prioritise inspection of {abnormal} cycles with model-detected abnormal resistance.")
    else:
        st.success("No abnormal-resistance cycles were identified by the model.")
    if detail["ambiguous_operations"]:
        st.warning(f"{detail['ambiguous_operations']} cycles have ambiguous operation flags. The fitted global threshold was used.")
    display = cycles.copy()
    display["Cycle"] = range(1, len(display)+1)
    display["Start"] = pd.to_datetime(display.start_time.map(timestamp), unit="s", utc=True)
    display["End"] = pd.to_datetime(display.end_time.map(timestamp), unit="s", utc=True)
    st.subheader("Door-cycle timeline")
    chart(px.timeline(display, x_start="Start", x_end="End", y="operation", color="prediction",
          color_discrete_map=COLOURS, hover_data=["Cycle", "current_sum", "threshold"],
          labels={"operation": "Operation", "prediction": "Condition"}), 265)
    st.dataframe(display[["Cycle", "start_time", "end_time", "operation", "prediction"]].rename(
        columns={"start_time":"Start time","end_time":"End time","operation":"Operation","prediction":"Condition"}),
        hide_index=True, width="stretch")
    with st.expander("Input and cycle diagnostics"):
        st.caption(f"{detail['input_rows']:,} readings. Cycle split: timestamp gap greater than {detail['gap_threshold_s']} second.")
        st.caption("Current sum is measured in mA samples. The reference threshold is fitted from training data.")
        st.dataframe(display[["Cycle","operation","n_rows","duration_s","current_sum","threshold","operation_ambiguous"]],
                     hide_index=True, width="stretch")


def show_acv(result):
    selected = st.selectbox("Case", result.source_files) if len(result.source_files)>1 else result.source_files[0]
    detail = result.diagnostics["by_file"][selected]
    ranking = pd.DataFrame(detail["ranking"])
    top = ranking.iloc[0]
    cols = st.columns([1.4,1,1])
    cols[0].metric("Highest priority inspection", f"CAR {top['car']}", border=True)
    cols[1].metric("Top-two score separation", f"{detail['score_margin']:.3f}", border=True)
    cols[2].metric("Methods selecting this car first", f"{int(top['methods_ranking_car_first'])} / {detail['method_count']}", border=True)
    st.caption("Model-ranked inspection priority. Scores are relative suspicion scores, not failure probabilities.")
    graph = ranking.iloc[::-1].copy()
    graph["Car"] = "Car " + graph.car
    fig = px.bar(graph, x="ensemble_score", y="Car", orientation="h",
        color=graph.car.eq(top["car"]).map({True:"Highest priority",False:"Other cars"}),
        color_discrete_map={"Highest priority":"#D57729","Other cars":"#3C6B8A"},
        labels={"ensemble_score":"Suspicion score"})
    fig.update_layout(showlegend=False)
    fig.update_xaxes(range=[0,1.05])
    chart(fig,345)
    st.dataframe(ranking[["predicted_rank","car","ensemble_score","methods_ranking_car_first"]].rename(
        columns={"predicted_rank":"Rank","car":"Car","ensemble_score":"Suspicion score","methods_ranking_car_first":"Methods selecting car first"}),
        hide_index=True,width="stretch")
    with st.expander("Thermal and component diagnostics"):
        st.caption(f"{detail['input_rows']:,} readings. Car identifiers come from this file's headers.")
        st.dataframe(pd.DataFrame(detail["features"])[["car","valid_fraction","indoor_mean","cooling_gap_mean","rel_indoor_mean","rel_gap_mean"]],
                     hide_index=True,width="stretch")
        st.dataframe(ranking,hide_index=True,width="stretch")


def show_rail(result):
    for col,label in zip(st.columns(3),["Normal","Side I","Side II"]):
        col.metric(label,int(result.table.prediction.eq(label).sum()),border=True)
    if len(result.table)==1:
        label=result.table.iloc[0].prediction
        st.subheader(f"Model result: {label}")
        st.write("No corrugation signature was identified by the model." if label=="Normal" else
                 f"The model detected a corrugation signature associated with {label}. Prioritise inspection of the corresponding rail side.")
    else:
        counts=result.table.prediction.value_counts().reindex(["Normal","Side I","Side II"],fill_value=0).rename_axis("Condition").reset_index(name="Files")
        fig=px.bar(counts,x="Condition",y="Files",color="Condition",color_discrete_map=COLOURS)
        fig.update_layout(showlegend=False)
        chart(fig,230)
    low=[entry for d in result.diagnostics["by_file"].values() for entry in d.get("low_speed_rule",[])]
    if low:
        st.info(f"{len(low)} recording(s) returned Normal under the low-speed domain rule. The classifier was not invoked for them.")
    st.dataframe(result.table.rename(columns={"file_id":"Recording","prediction":"Model result"}),hide_index=True,width="stretch")
    st.caption("These are model results, not confirmed physical defects.")
    with st.expander("Input diagnostics"):
        st.write(f"{len(result.source_files)} files analysed successfully; no processing fallbacks entered the output.")
        if low:
            st.dataframe(pd.DataFrame(low),hide_index=True,width="stretch")


def show_shm(result):
    values=result.table
    if len(values)==1:
        st.metric("Predicted cumulative fatigue damage",f"{values.iloc[0].prediction:.6g}",border=True)
    else:
        st.metric("Stress recordings analysed",len(values),border=True)
        chart(px.bar(values,x="file_id",y="prediction",labels={"file_id":"Recording","prediction":"Cumulative fatigue damage"},
                     color_discrete_sequence=["#3C6B8A"]),280)
    st.caption("Model estimate for each recording segment. Remaining useful life is not inferred.")
    st.dataframe(values.rename(columns={"file_id":"Recording","prediction":"Predicted damage"}),hide_index=True,width="stretch")
    with st.expander("Stress trace and input diagnostics"):
        name=st.selectbox("Stress recording",result.source_files)
        d=result.diagnostics["by_file"][name]["files"][0]
        st.write(f"{d['input_rows']:,} readings; {d['counted_cycles']:,} counted cycles.")
        chart(px.line(x=d["trace_index"],y=d["trace"],labels={"x":"Sample index","y":"Stress"},
                      color_discrete_sequence=["#3C6B8A"]),250)


RENDERERS={"door":show_door,"acv":show_acv,"rail":show_rail,"shm":show_shm}
