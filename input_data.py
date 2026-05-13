import streamlit as st
import pandas as pd
import numpy as np
import io
import hashlib
from ui_components import render_info_panel, fix_arrow_compatibility


def _clear_derived_keys():
    """Clear all derived session state keys while preserving a minimal set.

    Keeps only `global_seed` and `page`. This avoids keeping duplicate large
    DataFrame copies in session state when a new file is uploaded.
    """
    keep_keys = {"global_seed", "page"}
    for k in list(st.session_state.keys()):
        if k in keep_keys:
            continue
        try:
            del st.session_state[k]
        except Exception:
            st.session_state[k] = None


def render_input_data():
    st.header("📊 Data")
    st.write("Upload a train CSV file with header. Uploading a new file resets derived data.")

    uploaded_file = st.file_uploader("Upload Train CSV (with header)", type=["csv"])

    if uploaded_file is not None:
        try:
            file_bytes = uploaded_file.read()
            df = pd.read_csv(io.BytesIO(file_bytes)).reset_index(drop=True)

            # Clean any existing index columns
            if "Unnamed: 0" in df.columns:
                df = df.drop(columns=["Unnamed: 0"])

        except pd.errors.EmptyDataError:
            st.error("The uploaded file is empty. Please upload a valid CSV file.")
            return
        except pd.errors.ParserError:
            st.error("Error parsing CSV file. Please upload a valid CSV file.")
            return
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            return

        # Detect new uploads via hash and clear stale derived state
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        if st.session_state.get("upload_hash") != file_hash:
            _clear_derived_keys()
            st.session_state["upload_hash"] = file_hash

        # --- Preview & configuration ---
        st.subheader("Data Preview")
        st.dataframe(fix_arrow_compatibility(df.head(10)), use_container_width=True, hide_index=True)

        target_col = st.selectbox("Select Target Column", options=list(df.columns), key="target_col_select")
        task_type = st.radio("Task Type", ["Classification", "Regression"])

        if st.button("✅ Confirm & Proceed"):
            st.session_state["df"] = df.copy()
            st.session_state["target_column"] = target_col
            st.session_state["task_type"] = task_type
            st.success("Dataset saved to session. You can proceed to EDA.")

        # --- Data insight tabs (always visible after upload) ---
        _render_data_tabs(df)

    else:
        if st.session_state.get("df") is not None:
            df_confirmed = st.session_state["df"]

            st.subheader("Current dataset (in session)")
            st.dataframe(fix_arrow_compatibility(df_confirmed.head(10)), use_container_width=True, hide_index=True)
            st.markdown(f"**Target column:** {st.session_state.get('target_column')}  ")
            st.markdown(f"**Task type:** {st.session_state.get('task_type')}  ")
            st.write("To replace this dataset, upload a new CSV above or click Replace Dataset.")

            # --- Data insight tabs for confirmed dataset ---
            _render_data_tabs(df_confirmed)

            if st.button("🔁 Replace Dataset"):
                _clear_derived_keys()
                st.success("Ready for new dataset upload.")
                try:
                    st.rerun()
                except AttributeError:
                    try:
                        st.experimental_rerun()
                    except Exception:
                        pass
        else:
            st.info("No file uploaded yet. Use the uploader above to upload a CSV file.")

    render_info_panel("Data")


def _render_data_tabs(df: pd.DataFrame):
    """Render the four data-insight tabs for a given DataFrame."""
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Complete Table", "☣️ Null Values", "🎌 Duplicated Values", "📑 Data Types"]
    )

    # Tab 1 — Complete Table
    with tab1:
        st.subheader("Complete Table")
        st.dataframe(fix_arrow_compatibility(df), use_container_width=True, hide_index=True)

    # Tab 2 — Null Values
    with tab2:
        st.subheader("Null Values")
        null_counts = df.isnull().sum().reset_index()
        null_counts.columns = ["Column name", "Missing value count"]
        st.dataframe(null_counts, use_container_width=True, hide_index=True)

    # Tab 3 — Duplicated Values
    with tab3:
        st.subheader("Duplicated Values")
        dup_count = df.duplicated().sum()
        st.write(f"Total duplicated rows: **{dup_count}**")
        if dup_count > 0:
            st.dataframe(
                fix_arrow_compatibility(df[df.duplicated()]),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.success("No duplicated rows found.")

    # Tab 4 — Data Types
    with tab4:
        st.subheader("Data Types Summary")
        type_info = pd.DataFrame(
            {"Column": df.columns, "Type": df.dtypes.values}
        ).reset_index(drop=True)
        st.dataframe(type_info, use_container_width=True, hide_index=True)
