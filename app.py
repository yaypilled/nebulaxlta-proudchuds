"""Four trained models in one maintenance wrapper."""
import logging
import streamlit as st
from src.app.components import INFO, RENDERERS
from src.app.maintenance import VISUAL_CSS, NAMES, findings, summary_html
from src.app.service import SUBSYSTEMS, analyse, model_status, source_digest
from src.app.submission import predictions_zip

logging.basicConfig(level=logging.INFO)
st.set_page_config(page_title="Train checks", page_icon=":material/train:", layout="wide")
st.markdown("""<style>
.stApp{background:#F7F8FA;color:#182B3A}
.block-container{padding-top:2rem;padding-bottom:3rem;max-width:1180px}
h1{font-size:2rem!important;letter-spacing:-.6px}
[data-baseweb="tab-list"]{gap:8px;border-bottom:1px solid #D0D5DD;padding-bottom:10px}
button[data-baseweb="tab"]{min-height:52px;padding:0 20px;background:#fff;border:1px solid #D0D5DD;border-radius:8px;color:#344054;font-weight:600}
button[data-baseweb="tab"][aria-selected="true"]{background:#174C47;color:#fff;border-color:#174C47}
button:focus-visible,summary:focus-visible{outline:3px solid #1570EF!important;outline-offset:3px}
[data-testid="stFileUploader"]{background:white;border-radius:10px}
[data-testid="stMetric"]{background:white;border-radius:8px;padding:12px}
@media(max-width:640px){button[data-baseweb="tab"]{padding:0 12px}.block-container{padding-left:1rem;padding-right:1rem}}
""" + VISUAL_CSS + "</style>", unsafe_allow_html=True)


@st.cache_resource
def backend_status():
    return {key: model_status(key) for key in SUBSYSTEMS}


status = backend_status()
results = st.session_state.setdefault("results", {})
st.title("Train checks")
st.caption("Choose a subsystem · Upload data · Check results")

# All four upload widgets stay mounted so changing tabs preserves selections.
for tab, key in zip(st.tabs([NAMES[k] for k in SUBSYSTEMS]), SUBSYSTEMS):
    with tab:
        if not status[key][0]:
            st.error("This check is temporarily unavailable. Please contact the app administrator.")
            results.pop(key, None)
            continue
        st.write(INFO[key]["input"])
        uploads = st.file_uploader(f"{NAMES[key]} data", type=["xlsx"] if key == "acv" else ["csv"],
            accept_multiple_files=key != "door", key=f"uploads_{key}", label_visibility="collapsed", max_upload_size=64)
        uploads = ([uploads] if uploads is not None else []) if key == "door" else (uploads or [])
        names = [upload.name for upload in uploads]
        duplicates = len(names) != len(set(names))
        files = {upload.name: upload.getvalue() for upload in uploads}
        changed = key in results and (duplicates or not files or source_digest(files) != results[key].source_digest)
        if changed:
            results.pop(key, None)
            if files:
                st.info("Files changed. Run the check again to update the result.")
        if duplicates:
            st.error("Two files have the same name. Remove or rename the duplicate.")
        if st.button("Check data", type="primary", icon=":material/play_arrow:", key=f"analyse_{key}", disabled=not files or duplicates):
            results.pop(key, None)
            bar = st.progress(0.0, text="Checking your data…")
            try:
                def update(current, total, name):
                    bar.progress(current/total, text=f"Checked {current} of {total}: {name}")
                results[key] = analyse(key, files, progress=update)
            except ValueError as exc:
                st.error(str(exc))
            finally:
                bar.empty()
        if key in results:
            result = results[key]
            st.caption("Checked: " + ", ".join(result.source_files))
            RENDERERS[key](result)
            st.download_button("Download results (CSV)", result.csv_bytes, result.filename, "text/csv",
                key=f"download_{key}", icon=":material/download:")
        del files, uploads

if results:
    st.divider()
    st.subheader("Summary report")
    st.caption(f"{len(results)} of 4 subsystems checked. The report includes current results from all tabs; unchecked systems are labelled.")
    with st.expander("Preview summary"):
        for key in SUBSYSTEMS:
            st.write(f"**{NAMES[key]}**")
            if key in results:
                for row in findings(results[key]):
                    st.text(f"{row['source']} · {row['status']}\n{row['finding']}\n{row['action']}")
            else:
                st.write("Not checked")
    st.download_button("Download summary report", summary_html(results), "maintenance_summary.html", "text/html",
        type="primary", key="summary_report", icon=":material/download:")
    st.caption("Open the report to read or print it. Choose Print → Save as PDF for a PDF copy.")
    with st.expander("Download all prediction files"):
        st.caption("Includes " + ", ".join(result.filename for result in results.values()))
        st.download_button("Download predictions.zip", predictions_zip(results), "predictions.zip", "application/zip", key="predictions_zip")
