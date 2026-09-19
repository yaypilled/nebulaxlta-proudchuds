"""Four trained models in one maintenance wrapper."""
import logging
import streamlit as st
from src.app.components import INFO, RENDERERS
from src.app.maintenance import VISUAL_CSS, NAMES, findings, summary_html
from src.app.service import SUBSYSTEMS, analyse, model_status, source_digest
from src.app.submission import predictions_zip

logging.basicConfig(level=logging.INFO)
st.set_page_config(page_title="Train checks", page_icon=":material/train:", layout="wide")

# One design system, defined once. Colours are tokens so the accent can be
# changed in a single place; every status colour is paired with a text label
# and a glyph elsewhere in the app, so colour is never the only signal.
st.markdown("""<style>
:root{
  --ink:#101828; --ink-soft:#475467; --ink-faint:#667085;
  --line:#E4E7EC; --line-strong:#D0D5DD;
  --surface:#FFFFFF; --canvas:#F7F8FA;
  --accent:#174C47; --accent-hover:#12403C;
  --focus:#1570EF;
  --radius:10px; --radius-sm:8px;
  --shadow:0 1px 2px rgba(16,24,40,.05);
  --shadow-lift:0 4px 12px rgba(16,24,40,.08);
}
.stApp{background:var(--canvas);color:var(--ink)}
.block-container{padding-top:2.25rem;padding-bottom:4rem;max-width:1100px}

/* Typography ---------------------------------------------------------- */
h1{font-size:1.9rem!important;letter-spacing:-.5px;font-weight:700;margin-bottom:.15rem!important}
h2,h3{letter-spacing:-.2px}
.stCaption,[data-testid="stCaptionContainer"]{color:var(--ink-faint)}

/* Tabs: quieter until selected, so the eye goes to content ------------- */
[data-baseweb="tab-list"]{gap:6px;border-bottom:1px solid var(--line);padding-bottom:12px;margin-bottom:6px}
button[data-baseweb="tab"]{
  min-height:46px;padding:0 18px;background:var(--surface);
  border:1px solid var(--line-strong);border-radius:var(--radius-sm);
  color:var(--ink-soft);font-weight:600;font-size:.94rem;
  transition:border-color .15s ease,color .15s ease,box-shadow .15s ease;
}
button[data-baseweb="tab"]:hover{border-color:var(--accent);color:var(--accent)}
button[data-baseweb="tab"][aria-selected="true"]{
  background:var(--accent);color:#fff;border-color:var(--accent);box-shadow:var(--shadow)
}
[data-baseweb="tab-highlight"],[data-baseweb="tab-border"]{display:none}

/* Cards: uploader, metrics, expanders share one surface treatment ------ */
[data-testid="stFileUploader"]{
  background:var(--surface);border:1px solid var(--line-strong);
  border-radius:var(--radius);padding:.5rem;box-shadow:var(--shadow)
}
[data-testid="stFileUploader"] section{border:none;background:transparent}
[data-testid="stMetric"]{
  background:var(--surface);border:1px solid var(--line);
  border-radius:var(--radius-sm);padding:14px 16px;box-shadow:var(--shadow)
}
[data-testid="stExpander"]{
  background:var(--surface);border:1px solid var(--line);
  border-radius:var(--radius-sm);box-shadow:var(--shadow)
}
[data-testid="stExpander"] summary{font-weight:600;color:var(--ink-soft)}

/* Buttons -------------------------------------------------------------- */
.stButton>button,.stDownloadButton>button{
  border-radius:var(--radius-sm);font-weight:600;
  transition:transform .08s ease,box-shadow .15s ease,background .15s ease
}
.stButton>button[kind="primary"],.stDownloadButton>button[kind="primary"]{
  background:var(--accent);border-color:var(--accent)
}
.stButton>button[kind="primary"]:hover:not(:disabled),
.stDownloadButton>button[kind="primary"]:hover:not(:disabled){
  background:#12403C;border-color:#12403C;box-shadow:var(--shadow-lift)
}
.stButton>button:active:not(:disabled){transform:translateY(1px)}
.stButton>button:disabled{opacity:.5}

/* Alerts: flatter, less shouty than the default -------------------------*/
[data-testid="stAlert"]{border-radius:var(--radius-sm);border-width:1px;box-shadow:none}

/* Dividers and progress ------------------------------------------------ */
hr{border-color:var(--line)!important;margin:2rem 0 1.5rem}
[data-testid="stProgressBar"]>div>div>div{background:var(--accent)}

/* Accessibility: a visible focus ring on every interactive element ----- */
button:focus-visible,summary:focus-visible,[data-baseweb="tab"]:focus-visible,
input:focus-visible,[data-testid="stFileUploader"] *:focus-visible{
  outline:3px solid var(--focus)!important;outline-offset:2px;border-radius:4px
}

/* Respect a reduced-motion preference ---------------------------------- */
@media(prefers-reduced-motion:reduce){
  *{transition:none!important;animation:none!important}
}

/* Phone ---------------------------------------------------------------- */
@media(max-width:640px){
  .block-container{padding-left:1rem;padding-right:1rem;padding-top:1.5rem}
  h1{font-size:1.5rem!important}
  button[data-baseweb="tab"]{padding:0 12px;min-height:42px;font-size:.88rem}
  [data-baseweb="tab-list"]{overflow-x:auto;flex-wrap:nowrap}
}
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
        # predictions_zip validates every result and raises on a bad one. Build
        # it before rendering the control: calling it inside st.download_button
        # lets the exception escape and blanks the whole page, and this is the
        # one download that matters most.
        try:
            archive = predictions_zip(results)
        except ValueError as exc:
            st.error(f"Cannot build predictions.zip: {exc}")
        else:
            st.download_button("Download predictions.zip", archive, "predictions.zip",
                "application/zip", key="predictions_zip", icon=":material/folder_zip:")
