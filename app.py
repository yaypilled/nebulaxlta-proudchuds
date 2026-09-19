"""Train Condition Intelligence: one upload-to-submission workflow for PS3."""
import logging
import streamlit as st
from src.app.components import INFO, RENDERERS, explain
from src.app.service import SUBSYSTEMS, analyse, model_status, source_digest
from src.app.submission import predictions_zip

logging.basicConfig(level=logging.INFO)
st.set_page_config(page_title="Train Condition Intelligence",page_icon=":material/train:",
                   layout="wide",initial_sidebar_state="expanded")
st.markdown("""
<style>
.stApp{background:#F5F7FA;color:#20344C}
[data-testid="stSidebar"]{background:#11283E}
[data-testid="stSidebar"] *{color:#E8EEF5}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] *{color:#A8BCCC}
.block-container{padding-top:2.3rem;padding-bottom:3rem;max-width:1440px}
h1{font-weight:700;letter-spacing:-1.1px} h2,h3{letter-spacing:-.3px}
[data-testid="stMetric"]{background:white;border-color:#DEE5ED;border-radius:10px;padding:18px}
[data-testid="stFileUploader"],[data-testid="stExpander"]{background:white;border-radius:10px}
div.stButton>button[kind="primary"],div.stDownloadButton>button[kind="primary"]{background:#176B75;border-color:#176B75}
.eyebrow{font-size:11px;font-weight:700;letter-spacing:2px;color:#55778F;margin-bottom:10px}
.hero-note{color:#63768A;font-size:16px;max-width:780px;line-height:1.6}
</style>
""",unsafe_allow_html=True)


@st.cache_resource
def backend_status():
    return {key:model_status(key) for key in SUBSYSTEMS}


def go(page):
    st.session_state["page"]=page


status=backend_status()
st.session_state.setdefault("results",{})
st.session_state.setdefault("page","Overview")
pages=["Overview",*[INFO[key]["title"] for key in SUBSYSTEMS],"Submission Centre"]
with st.sidebar:
    st.markdown("### TRAIN CONDITION\n### INTELLIGENCE")
    st.caption("LTA × NebulaX  |  PS3")
    st.divider()
    st.radio("Workspace",pages,key="page",label_visibility="collapsed")
    st.divider()
    st.caption("MODEL AVAILABILITY")
    for key in SUBSYSTEMS:
        st.caption(f"{INFO[key]['title']} · {status[key][1]}")
    st.divider()
    st.caption(f"{len(st.session_state['results'])} of 4 prediction outputs generated")
    st.caption("Engineering decision support")
page=st.session_state["page"]
results=st.session_state["results"]

if page=="Overview":
    st.markdown('<div class="eyebrow">RAIL CONDITION MONITORING</div>',unsafe_allow_html=True)
    st.title("From sensor data to inspection priorities.")
    st.markdown('<div class="hero-note">Analyse train telemetry, inspect the model’s findings and download validated prediction files from one workspace.</div>',unsafe_allow_html=True)
    st.write("")
    cols=st.columns(3)
    cols[0].metric("Operational subsystems",f"{sum(ready for ready,_ in status.values())} / 4",border=True)
    cols[1].metric("Outputs generated",f"{len(results)} / 4",border=True)
    cols[2].metric("Official output format","CSV + ZIP",border=True)
    st.write("")
    st.subheader("Choose a subsystem")
    for pair in [SUBSYSTEMS[:2],SUBSYSTEMS[2:]]:
        for column,key in zip(st.columns(2),pair):
            info=INFO[key]
            with column,st.container(border=True):
                st.caption(info["tag"])
                st.subheader(info["title"])
                st.write(info["summary"])
                st.caption(info["result"])
                st.button(f"Open {info['title']}",key=f"open_{key}",on_click=go,args=(info["title"],),
                          disabled=not status[key][0],width="stretch")
    st.info("Upload the held-out test inputs in each subsystem, then collect the official CSVs in the Submission Centre.")

elif page=="Submission Centre":
    st.markdown('<div class="eyebrow">OFFICIAL PREDICTION OUTPUTS</div>',unsafe_allow_html=True)
    st.title("Submission Centre")
    st.write("Download predictions generated in this session. The archive contains only official prediction CSVs at its root.")
    for key in SUBSYSTEMS:
        with st.container(border=True):
            left,right=st.columns([2,1])
            left.subheader(INFO[key]["title"])
            if key in results:
                result=results[key]
                errors=result.validation_errors()
                if errors:
                    left.error("Validation failed: "+" ".join(errors))
                else:
                    left.caption(f"READY · {len(result.table)} prediction rows from {len(result.source_files)} file(s)")
                    right.download_button("Download CSV",result.csv_bytes,result.filename,"text/csv",
                                          key=f"submit_{key}",width="stretch")
                    with left.expander("Source files"):
                        st.write(", ".join(result.source_files))
            else:
                left.caption("NOT GENERATED")
                right.button("Analyse data",key=f"start_{key}",on_click=go,args=(INFO[key]["title"],),
                             disabled=not status[key][0],width="stretch")
    if results:
        try:
            archive=predictions_zip(results)
            st.download_button("Download predictions.zip",archive,"predictions.zip","application/zip",
                               type="primary",icon=":material/download:",width="stretch")
            st.caption("Includes "+", ".join(result.filename for result in results.values()))
        except ValueError as exc:
            st.error(str(exc))
    else:
        st.info("Analyse at least one subsystem to create predictions.zip.")
    st.caption("READY means schema and upload-coverage checks passed. Confirm the source files are the official held-out inputs before submitting.")

else:
    key=next(key for key in SUBSYSTEMS if INFO[key]["title"]==page)
    info=INFO[key]
    st.markdown(f'<div class="eyebrow">{info["tag"]}</div>',unsafe_allow_html=True)
    st.title(info["title"])
    st.write(info["summary"])
    if not status[key][0]:
        st.error("This subsystem is unavailable because its fitted model could not be loaded.")
        st.stop()
    with st.container(border=True):
        st.subheader("1. Upload telemetry")
        st.caption(info["input"])
        uploads=st.file_uploader("Telemetry files",type=["xlsx"] if key=="acv" else ["csv"],
                                accept_multiple_files=key!="door",key=f"uploads_{key}",
                                label_visibility="collapsed",max_upload_size=64)
        uploads=([uploads] if uploads is not None else []) if key=="door" else (uploads or [])
        names=[upload.name for upload in uploads]
        duplicates=len(names)!=len(set(names))
        if duplicates:
            st.error("Two files have the same name. Remove the duplicate before analysis.")
            results.pop(key,None)
        files={upload.name:upload.getvalue() for upload in uploads}
        if files and key in results and source_digest(files)!=results[key].source_digest:
            results.pop(key,None)
            st.info("Uploaded files changed. Analyse this selection to generate a new output.")
        if st.button("Analyse",type="primary",icon=":material/analytics:",key=f"analyse_{key}",
                     disabled=not files or duplicates):
            results.pop(key,None)
            bar=st.progress(0.0,text="Checking telemetry and loading the fitted model…")
            try:
                def update(current,total,name):
                    bar.progress(current/total,text=f"Analysed {current}/{total}: {name}")
                results[key]=analyse(key,files,progress=update)
                st.success("Analysis complete. The official output passed validation.")
            except ValueError as exc:
                st.error(str(exc))
            finally:
                bar.empty()
    if key in results:
        result=results[key]
        st.subheader("2. Inspect the result")
        st.caption(f"Analysed {', '.join(result.source_files)} in {result.elapsed_seconds:.1f} seconds.")
        RENDERERS[key](result)
        st.write("")
        left,right=st.columns(2)
        left.download_button("Download official prediction CSV",result.csv_bytes,result.filename,"text/csv",
                             type="primary",icon=":material/download:",key=f"download_{key}",width="stretch")
        right.button("Open Submission Centre",on_click=go,args=("Submission Centre",),width="stretch")
    explain(key)
