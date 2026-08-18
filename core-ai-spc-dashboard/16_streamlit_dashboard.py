import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from pathlib import Path


# ============================================================
# 0. PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI-SPC Process Monitor",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 1. PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TABLE_DIR = (
    BASE_DIR
    / "results"
    / "tables"
)

SYSTEM_FILE = (
    TABLE_DIR
    / "final_ai_spc_system.csv"
)

METRICS_FILE = (
    TABLE_DIR
    / "final_evaluation_metrics.csv"
)

FIRST_ALARM_FILE = (
    TABLE_DIR
    / "final_first_alarm_performance.csv"
)

THRESHOLD_FILE = (
    TABLE_DIR
    / "final_threshold_analysis.csv"
)

PROCESS_IMPORTANCE_FILE = (
    TABLE_DIR
    / "shap_process_interpretation.csv"
)


# ============================================================
# 2. CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

/* ----------------------------------------------------------
   전체 폭
---------------------------------------------------------- */

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1500px;
}


/* ----------------------------------------------------------
   타이틀
---------------------------------------------------------- */

.main-title {
    font-size: 2.6rem;
    font-weight: 800;
    margin-bottom: 0.1rem;
    color: #1f2937;
}

.sub-title {
    color: #6b7280;
    font-size: 1.05rem;
    margin-bottom: 1.8rem;
}


/* ----------------------------------------------------------
   카드
---------------------------------------------------------- */

.monitor-card {
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 1.3rem 1.4rem;
    background-color: #ffffff;
    min-height: 205px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}

.card-title {
    font-size: 0.92rem;
    color: #6b7280;
    font-weight: 600;
    margin-bottom: 0.45rem;
}

.card-main {
    font-size: 2rem;
    font-weight: 800;
    color: #111827;
    line-height: 1.15;
    margin-bottom: 0.8rem;
}

.card-sub {
    font-size: 0.98rem;
    color: #374151;
    line-height: 1.55;
}

.card-caption {
    color: #9ca3af;
    font-size: 0.85rem;
    margin-top: 0.6rem;
}


/* ----------------------------------------------------------
   상태 badge
---------------------------------------------------------- */

.badge-normal {
    display: inline-block;
    padding: 0.45rem 0.85rem;
    border-radius: 999px;
    background-color: #dcfce7;
    color: #166534;
    font-weight: 700;
    font-size: 1rem;
}

.badge-warning {
    display: inline-block;
    padding: 0.45rem 0.85rem;
    border-radius: 999px;
    background-color: #ffedd5;
    color: #c2410c;
    font-weight: 700;
    font-size: 1rem;
}

.badge-alarm {
    display: inline-block;
    padding: 0.45rem 0.85rem;
    border-radius: 999px;
    background-color: #fee2e2;
    color: #b91c1c;
    font-weight: 700;
    font-size: 1rem;
}


/* ----------------------------------------------------------
   SHAP contributor 카드
---------------------------------------------------------- */

.shap-card {
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 1rem 1.1rem;
    min-height: 150px;
    background-color: #f9fafb;
}

.shap-rank {
    font-size: 0.85rem;
    font-weight: 800;
    color: #6b7280;
}

.shap-name {
    font-size: 1.02rem;
    font-weight: 700;
    margin-top: 0.25rem;
    margin-bottom: 0.6rem;
    color: #1f2937;
}

.shap-value {
    font-size: 0.92rem;
    color: #4b5563;
    margin-bottom: 0.25rem;
}


/* ----------------------------------------------------------
   섹션 제목
---------------------------------------------------------- */

.section-title {
    font-size: 1.45rem;
    font-weight: 800;
    color: #1f2937;
    margin-top: 1.2rem;
    margin-bottom: 0.8rem;
}


/* ----------------------------------------------------------
   retrospective box
---------------------------------------------------------- */

.validation-box {
    border: 1px solid #dbeafe;
    border-radius: 14px;
    padding: 1.2rem;
    background-color: #eff6ff;
}


/* ----------------------------------------------------------
   하단 disclaimer
---------------------------------------------------------- */

.disclaimer {
    font-size: 0.84rem;
    color: #6b7280;
    line-height: 1.5;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# 3. DATA LOAD
# ============================================================

@st.cache_data
def load_data():

    system_df = pd.read_csv(
        SYSTEM_FILE
    )

    metrics_df = pd.read_csv(
        METRICS_FILE
    )

    first_alarm_df = pd.read_csv(
        FIRST_ALARM_FILE
    )

    threshold_df = pd.read_csv(
        THRESHOLD_FILE
    )

    process_importance_df = pd.read_csv(
        PROCESS_IMPORTANCE_FILE
    )


    # --------------------------------------------------------
    # bool column normalize
    # --------------------------------------------------------

    bool_columns = [
        "ai_warning",
        "successful_first_alarm_one_step_warning"
    ]


    for column in bool_columns:

        if column in system_df.columns:

            system_df[column] = (
                system_df[column]
                .astype(str)
                .str.lower()
                .map(
                    {
                        "true": True,
                        "false": False
                    }
                )
            )


    return (
        system_df,
        metrics_df,
        first_alarm_df,
        threshold_df,
        process_importance_df
    )


(
    system_df,
    metrics_df,
    first_alarm_df,
    threshold_df,
    process_importance_df
) = load_data()


# ============================================================
# 4. HELPERS
# ============================================================

def get_metric(
    metric_name,
    default=np.nan
):

    row = (
        metrics_df[
            metrics_df[
                "metric"
            ]
            == metric_name
        ]
    )

    if len(row) == 0:
        return default

    return (
        row
        .iloc[0][
            "value"
        ]
    )


def clean_process_name(
    name
):

    if pd.isna(name):
        return "-"

    clean = str(name)

    replacements = {
        "HeliumBPFlow":
            "Helium BP Flow",

        "HeliumBPPressure":
            "Helium BP Pressure",

        "PlatenRFReflectedPower":
            "Platen RF Reflected Power",

        "PlatenRFTuningCapacitor":
            "Platen RF Tuning Capacitor",

        "PlatenRFLoadPower":
            "Platen RF Load Power",

        "Heater2Temp":
            "Heater 2 Temperature",

        "Gas7Flow":
            "Gas 7 Flow",

        "SourceRFReflectedPower":
            "Source RF Reflected Power",

        "SourceRFLoadPower":
            "Source RF Load Power",

        "Pressure":
            "Chamber Pressure"
    }

    for old, new in replacements.items():

        clean = clean.replace(
            old,
            new
        )

    clean = clean.replace(
        "_mean",
        " (mean)"
    )

    clean = clean.replace(
        "_std",
        " (std)"
    )

    clean = clean.replace(
        "_min",
        " (min)"
    )

    clean = clean.replace(
        "_max",
        " (max)"
    )

    return clean


def status_badge(
    status
):

    if status == "NORMAL":

        return (
            '<span class="badge-normal">'
            '● NORMAL'
            '</span>'
        )

    if status == "EARLY WARNING":

        return (
            '<span class="badge-warning">'
            '● EARLY WARNING'
            '</span>'
        )

    if status == "SPC ALARM":

        return (
            '<span class="badge-alarm">'
            '● SPC ALARM'
            '</span>'
        )

    return status


def result_badge(
    result
):

    if result == "TRUE EARLY WARNING":

        return (
            "✅ TRUE EARLY WARNING"
        )

    if result == "FALSE EARLY WARNING":

        return (
            "⚠️ FALSE EARLY WARNING"
        )

    if result == "MISSED EARLY WARNING":

        return (
            "❌ MISSED EARLY WARNING"
        )

    if result == "CORRECT NORMAL":

        return (
            "✅ CORRECT NORMAL"
        )

    return str(result)


# ============================================================
# 5. SIDEBAR
# ============================================================

st.sidebar.markdown(
    "## AI-SPC Monitor"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Live Monitor",
        "Lot Timeline",
        "Model Performance",
        "Process Explanation"
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
**System concept**

Current wafer  
→ AI next-wafer prediction  
→ Process explanation  
→ Engineer review
"""
)

st.sidebar.caption(
    "Prediction horizon: exactly 1 wafer ahead"
)

st.sidebar.caption(
    "Historical-data decision-support prototype"
)


# ============================================================
# 6. HEADER
# ============================================================

st.markdown(
    """
<div class="main-title">
AI–SPC Process Monitor
</div>

<div class="sub-title">
AI-assisted one-wafer-ahead process drift early-warning prototype
</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# PAGE 1. LIVE MONITOR
# ============================================================

if page == "Live Monitor":

    # ========================================================
    # Selector
    # ========================================================

    selector_col1, selector_col2, selector_col3 = (
        st.columns(
            [
                1,
                1,
                2
            ]
        )
    )


    lot_list = sorted(
        system_df[
            "lot_number"
        ]
        .unique()
    )


    with selector_col1:

        selected_lot = st.selectbox(
            "Lot",
            lot_list
        )


    lot_df = (
        system_df[
            system_df[
                "lot_number"
            ]
            == selected_lot
        ]
        .sort_values(
            "current_wafer"
        )
        .copy()
    )


    wafer_list = (
        lot_df[
            "current_wafer"
        ]
        .astype(int)
        .tolist()
    )


    with selector_col2:

        selected_current_wafer = (
            st.selectbox(
                "Current wafer",
                wafer_list
            )
        )


    selected_row = (
        lot_df[
            lot_df[
                "current_wafer"
            ]
            == selected_current_wafer
        ]
        .iloc[0]
    )


    next_wafer = int(
        selected_row[
            "next_wafer"
        ]
    )


    with selector_col3:

        st.markdown(
            f"""
<div style="
padding-top:1.9rem;
font-size:1.15rem;
font-weight:650;
color:#374151;
">
Lot {int(selected_lot)} |
Current W{int(selected_current_wafer)}
→ Predict W{next_wafer}
</div>
""",
            unsafe_allow_html=True
        )


    st.markdown("---")


    # ========================================================
    # Top 3 status cards
    # ========================================================

    card1, card2, card3 = st.columns(
        3
    )


    current_spc_status = str(
        selected_row[
            "current_spc_status"
        ]
    )


    system_status = str(
        selected_row[
            "system_status"
        ]
    )


    engineer_action = str(
        selected_row[
            "engineer_action"
        ]
    )


    risk = float(
        selected_row[
            "predicted_risk"
        ]
    )


    # --------------------------------------------------------
    # Card 1 - Current SPC
    # --------------------------------------------------------

    with card1:

        st.markdown(
            f"""
<div class="monitor-card">

<div class="card-title">
1. CURRENT WAFER / CONVENTIONAL SPC
</div>

<div class="card-main">
{current_spc_status}
</div>

<div class="card-sub">
Current W{int(selected_current_wafer)} etch:
<b>{selected_row['current_mean_si_etch']:.3f}</b>
<br><br>
LCL:
<b>{selected_row['LCL']:.3f}</b>
&nbsp;&nbsp;
CL:
<b>{selected_row['CL']:.3f}</b>
&nbsp;&nbsp;
UCL:
<b>{selected_row['UCL']:.3f}</b>
</div>

<div class="card-caption">
Current wafer has already been measured.
</div>

</div>
""",
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # Card 2 - AI forecast
    # --------------------------------------------------------

    with card2:

        st.markdown(
            f"""
<div class="monitor-card">

<div class="card-title">
2. AI NEXT-WAFER FORECAST
</div>

<div class="card-main">
W{next_wafer} Risk: {risk:.1%}
</div>

<div style="margin-top:0.5rem;">
{status_badge(system_status)}
</div>

<div class="card-caption">
Warning threshold = 50%
<br>
Prediction horizon = 1 wafer
</div>

</div>
""",
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # Card 3 - Engineer
    # --------------------------------------------------------

    with card3:

        if system_status == "EARLY WARNING":

            action_text = (
                "ENGINEER REVIEW REQUIRED"
            )

            action_description = (
                "Review highlighted process signals "
                "before deciding whether intervention is needed."
            )

        elif system_status == "SPC ALARM":

            action_text = (
                "ENGINEER ACTION REQUIRED"
            )

            action_description = (
                "Conventional SPC has already detected "
                "an out-of-control wafer."
            )

        else:

            action_text = (
                "CONTINUE MONITORING"
            )

            action_description = (
                "No AI warning is issued at the current threshold."
            )


        st.markdown(
            f"""
<div class="monitor-card">

<div class="card-title">
3. ENGINEER DECISION SUPPORT
</div>

<div style="
font-size:1.35rem;
font-weight:800;
line-height:1.25;
color:#111827;
margin-bottom:0.8rem;
">
{action_text}
</div>

<div class="card-sub">
{action_description}
<br><br>
Conditioning:
<b>
{selected_row['conditioning_count']}
/
{selected_row['conditioning_surface']}
</b>
</div>

<div class="card-caption">
AI does not automatically change the recipe.
</div>

</div>
""",
            unsafe_allow_html=True
        )


    # ========================================================
    # WHY DID AI WARN?
    # ========================================================

    st.markdown(
        """
<div class="section-title">
Why did AI make this prediction?
</div>
""",
        unsafe_allow_html=True
    )


    shap_cols = st.columns(
        3
    )


    for rank in range(
        1,
        4
    ):

        process_name = (
            selected_row[
                f"top{rank}_process"
            ]
        )

        process_value = (
            selected_row[
                f"top{rank}_value"
            ]
        )

        process_shap_value = (
            selected_row[
                f"top{rank}_shap"
            ]
        )


        with shap_cols[
            rank - 1
        ]:

            if pd.isna(
                process_name
            ):

                st.markdown(
                    """
<div class="shap-card">
No positive process contributor
</div>
""",
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    f"""
<div class="shap-card">

<div class="shap-rank">
#{rank} PROCESS CONTRIBUTOR
</div>

<div class="shap-name">
{clean_process_name(process_name)}
</div>

<div class="shap-value">
Sensor summary value:
<b>{process_value:.4f}</b>
</div>

<div class="shap-value">
SHAP contribution:
<b>{process_shap_value:+.3f}</b>
</div>

</div>
""",
                    unsafe_allow_html=True
                )


    st.info(
        (
            "These SHAP contributors explain which process signals "
            "pushed the AI prediction toward higher SPC-violation risk. "
            "They are predictive associations, not proof of causal mechanism."
        )
    )


    # ========================================================
    # MINI SPC CHART
    # ========================================================

    st.markdown(
        """
<div class="section-title">
Current SPC Position
</div>
""",
        unsafe_allow_html=True
    )


    current_value = float(
        selected_row[
            "current_mean_si_etch"
        ]
    )


    spc_fig = go.Figure()


    # --------------------------------------------------------
    # Range
    # --------------------------------------------------------

    x_min = min(
        float(
            selected_row[
                "LCL"
            ]
        ),
        current_value
    ) - 0.20


    x_max = max(
        float(
            selected_row[
                "UCL"
            ]
        ),
        current_value
    ) + 0.20


    spc_fig.add_trace(
        go.Scatter(
            x=[
                selected_row[
                    "LCL"
                ],
                selected_row[
                    "UCL"
                ]
            ],

            y=[
                0,
                0
            ],

            mode="lines",

            line=dict(
                width=12
            ),

            name="Control range",

            hoverinfo="skip"
        )
    )


    spc_fig.add_trace(
        go.Scatter(
            x=[
                selected_row[
                    "LCL"
                ]
            ],

            y=[
                0
            ],

            mode="markers+text",

            marker=dict(
                size=14
            ),

            text=[
                "LCL"
            ],

            textposition=
                "bottom center",

            name="LCL"
        )
    )


    spc_fig.add_trace(
        go.Scatter(
            x=[
                selected_row[
                    "CL"
                ]
            ],

            y=[
                0
            ],

            mode="markers+text",

            marker=dict(
                size=14
            ),

            text=[
                "CL"
            ],

            textposition=
                "bottom center",

            name="CL"
        )
    )


    spc_fig.add_trace(
        go.Scatter(
            x=[
                selected_row[
                    "UCL"
                ]
            ],

            y=[
                0
            ],

            mode="markers+text",

            marker=dict(
                size=14
            ),

            text=[
                "UCL"
            ],

            textposition=
                "bottom center",

            name="UCL"
        )
    )


    spc_fig.add_trace(
        go.Scatter(
            x=[
                current_value
            ],

            y=[
                0
            ],

            mode="markers+text",

            marker=dict(
                size=22,
                symbol="diamond"
            ),

            text=[
                (
                    f"W{int(selected_current_wafer)}"
                    f"<br>{current_value:.3f}"
                )
            ],

            textposition=
                "top center",

            name="Current wafer"
        )
    )


    spc_fig.update_layout(
        height=250,

        margin=dict(
            l=20,
            r=20,
            t=20,
            b=20
        ),

        showlegend=False,

        xaxis=dict(
            title=
                "Mean Si etch depth",

            range=[
                x_min,
                x_max
            ]
        ),

        yaxis=dict(
            visible=False,
            range=[
                -0.5,
                0.5
            ]
        )
    )


    st.plotly_chart(
        spc_fig,
        use_container_width=True
    )


    # ========================================================
    # RETROSPECTIVE VALIDATION
    # ========================================================

    st.markdown("---")


    st.markdown(
        """
<div class="section-title">
Retrospective Validation
</div>

<div class="disclaimer">
This section contains information that would not be available
at the moment the AI prediction is issued.
It is shown only to evaluate whether the historical prediction
was correct.
</div>
""",
        unsafe_allow_html=True
    )


    show_outcome = st.toggle(
        "Reveal observed next-wafer outcome"
    )


    if show_outcome:

        outcome1, outcome2 = st.columns(
            [
                1,
                1
            ]
        )


        with outcome1:

            st.markdown(
                f"""
<div class="validation-box">

<b>Observed W{next_wafer}</b>

<br><br>

Actual etch:
<b>
{selected_row['next_actual_mean_si_etch']:.3f}
</b>

<br><br>

Actual SPC result:
<b>
{selected_row['next_actual_status']}
</b>

</div>
""",
                unsafe_allow_html=True
            )


        with outcome2:

            warning_result = (
                selected_row[
                    "warning_result"
                ]
            )


            st.markdown(
                f"""
<div class="validation-box">

<b>Prediction evaluation</b>

<br><br>

{result_badge(warning_result)}

<br><br>

AI risk at W{int(selected_current_wafer)}:
<b>{risk:.1%}</b>

</div>
""",
                unsafe_allow_html=True
            )


        # ----------------------------------------------------
        # Successful first alarm warning
        # ----------------------------------------------------

        if bool(
            selected_row[
                "successful_first_alarm_one_step_warning"
            ]
        ):

            st.success(
                (
                    f"Successful 1-wafer-ahead warning: "
                    f"the AI warned at W{int(selected_current_wafer)}, "
                    f"and W{next_wafer} became the first SPC alarm."
                )
            )


        # ====================================================
        # Before vs after SPC plot
        # ====================================================

        validation_fig = go.Figure()


        validation_fig.add_hline(
            y=float(
                selected_row[
                    "CL"
                ]
            ),
            line_dash="solid",
            annotation_text="CL"
        )


        validation_fig.add_hline(
            y=float(
                selected_row[
                    "LCL"
                ]
            ),
            line_dash="dash",
            annotation_text="LCL"
        )


        validation_fig.add_hline(
            y=float(
                selected_row[
                    "UCL"
                ]
            ),
            line_dash="dash",
            annotation_text="UCL"
        )


        validation_fig.add_trace(
            go.Scatter(
                x=[
                    f"W{int(selected_current_wafer)}",
                    f"W{next_wafer}"
                ],

                y=[
                    selected_row[
                        "current_mean_si_etch"
                    ],
                    selected_row[
                        "next_actual_mean_si_etch"
                    ]
                ],

                mode="lines+markers+text",

                text=[
                    "Current",
                    "Observed next"
                ],

                textposition=
                    "top center",

                name="Etch depth"
            )
        )


        validation_fig.update_layout(
            title=(
                "Current wafer vs observed next wafer"
            ),

            xaxis_title=
                "Wafer",

            yaxis_title=
                "Mean Si etch depth",

            height=380,

            showlegend=False
        )


        st.plotly_chart(
            validation_fig,
            use_container_width=True
        )


# ============================================================
# PAGE 2. LOT TIMELINE
# ============================================================

elif page == "Lot Timeline":

    st.markdown(
        """
<div class="section-title">
Lot Timeline
</div>
""",
        unsafe_allow_html=True
    )


    lot_list = sorted(
        system_df[
            "lot_number"
        ]
        .unique()
    )


    selected_lot = st.selectbox(
        "Select Lot",
        lot_list,
        key="timeline_lot"
    )


    lot_df = (
        system_df[
            system_df[
                "lot_number"
            ]
            == selected_lot
        ]
        .sort_values(
            "next_wafer"
        )
        .copy()
    )


    # ========================================================
    # Risk chart
    # ========================================================

    fig = go.Figure()


    fig.add_trace(
        go.Scatter(
            x=lot_df[
                "next_wafer"
            ],

            y=lot_df[
                "predicted_risk"
            ],

            mode=
                "lines+markers",

            name=
                "AI next-wafer risk",

            hovertemplate=(
                "Predicted W%{x}<br>"
                "Risk=%{y:.1%}"
                "<extra></extra>"
            )
        )
    )


    fig.add_hline(
        y=0.50,
        line_dash="dash",
        annotation_text="AI warning threshold"
    )


    warning_rows = (
        lot_df[
            lot_df[
                "ai_warning"
            ]
            == True
        ]
    )


    if len(
        warning_rows
    ) > 0:

        fig.add_trace(
            go.Scatter(
                x=warning_rows[
                    "next_wafer"
                ],

                y=warning_rows[
                    "predicted_risk"
                ],

                mode="markers",

                marker=dict(
                    size=14,
                    symbol="triangle-up"
                ),

                name="AI warning"
            )
        )


    violation_rows = (
        lot_df[
            lot_df[
                "next_spc_violation"
            ]
            == 1
        ]
    )


    if len(
        violation_rows
    ) > 0:

        fig.add_trace(
            go.Scatter(
                x=violation_rows[
                    "next_wafer"
                ],

                y=violation_rows[
                    "predicted_risk"
                ],

                mode="markers",

                marker=dict(
                    size=14,
                    symbol="x"
                ),

                name="Observed SPC violation"
            )
        )


    first_alarm_rows = (
        lot_df[
            lot_df[
                "next_first_spc_alarm"
            ]
            == 1
        ]
    )


    if len(
        first_alarm_rows
    ) > 0:

        first_alarm_wafer = int(
            first_alarm_rows[
                "next_wafer"
            ]
            .iloc[0]
        )


        fig.add_vline(
            x=first_alarm_wafer,
            line_dash="dot",
            annotation_text=(
                f"First SPC alarm W"
                f"{first_alarm_wafer}"
            )
        )


    fig.update_layout(
        title=(
            f"Lot {int(selected_lot)}"
            " — One-Step-Ahead AI Risk"
        ),

        xaxis_title=
            "Predicted next wafer",

        yaxis_title=
            "SPC violation risk",

        yaxis=dict(
            range=[
                0,
                1
            ],
            tickformat=".0%"
        ),

        hovermode="x unified",

        height=500
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # ========================================================
    # Timeline table
    # ========================================================

    display_df = (
        lot_df[
            [
                "current_wafer",
                "next_wafer",
                "current_spc_status",
                "predicted_risk",
                "system_status",
                "next_actual_status",
                "warning_result"
            ]
        ]
        .copy()
    )


    display_df[
        "predicted_risk"
    ] = (
        display_df[
            "predicted_risk"
        ]
        .map(
            lambda x:
                f"{x:.1%}"
        )
    )


    display_df = display_df.rename(
        columns={
            "current_wafer":
                "Current wafer",

            "next_wafer":
                "Predicted wafer",

            "current_spc_status":
                "Current SPC",

            "predicted_risk":
                "AI risk",

            "system_status":
                "AI-SPC status",

            "next_actual_status":
                "Observed next wafer",

            "warning_result":
                "Evaluation"
        }
    )


    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PAGE 3. MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.markdown(
        """
<div class="section-title">
Model Performance
</div>
""",
        unsafe_allow_html=True
    )


    precision = float(
        get_metric(
            "precision"
        )
    )

    recall = float(
        get_metric(
            "recall"
        )
    )

    specificity = float(
        get_metric(
            "specificity"
        )
    )

    roc_auc = float(
        get_metric(
            "roc_auc"
        )
    )

    false_warning_rate = float(
        get_metric(
            "false_warning_rate"
        )
    )

    first_warning_rate = float(
        get_metric(
            "first_spc_alarm_warning_rate"
        )
    )


    perf_cols = st.columns(
        6
    )


    metric_values = [
        (
            "Recall",
            f"{recall:.1%}"
        ),

        (
            "Precision",
            f"{precision:.1%}"
        ),

        (
            "Specificity",
            f"{specificity:.1%}"
        ),

        (
            "ROC-AUC",
            f"{roc_auc:.3f}"
        ),

        (
            "False Warning",
            f"{false_warning_rate:.1%}"
        ),

        (
            "First Alarm Warning",
            f"{first_warning_rate:.1%}"
        )
    ]


    for column, (
        label,
        value
    ) in zip(
        perf_cols,
        metric_values
    ):

        with column:

            st.metric(
                label,
                value
            )


    st.info(
        (
            "Among 8 observed first-SPC-alarm events, "
            "5 were correctly warned at the immediately preceding wafer "
            "(62.5%)."
        )
    )


    # ========================================================
    # First alarm risk
    # ========================================================

    first_display = (
        first_alarm_df[
            [
                "lot_number",
                "current_wafer",
                "next_wafer",
                "predicted_risk",
                "one_step_prediction"
            ]
        ]
        .copy()
    )


    first_display[
        "Result"
    ] = np.where(
        first_display[
            "one_step_prediction"
        ]
        == 1,

        "Warned",

        "Missed"
    )


    first_alarm_fig = go.Figure()


    first_alarm_fig.add_trace(
        go.Bar(
            x=[
                (
                    f"Lot "
                    f"{int(x)}"
                )
                for x
                in first_display[
                    "lot_number"
                ]
            ],

            y=first_display[
                "predicted_risk"
            ],

            text=[
                f"{x:.1%}"
                for x
                in first_display[
                    "predicted_risk"
                ]
            ],

            textposition="outside",

            name="Risk"
        )
    )


    first_alarm_fig.add_hline(
        y=0.50,
        line_dash="dash",
        annotation_text="Warning threshold"
    )


    first_alarm_fig.update_layout(
        title=
            "AI Risk Immediately Before the First SPC Alarm",

        yaxis=dict(
            range=[
                0,
                1
            ],
            tickformat=".0%"
        ),

        xaxis_title="Lot",

        yaxis_title="Predicted next-wafer risk",

        height=450
    )


    st.plotly_chart(
        first_alarm_fig,
        use_container_width=True
    )


    # ========================================================
    # Threshold trade-off
    # ========================================================

    st.markdown(
        """
<div class="section-title">
Threshold Trade-off
</div>
""",
        unsafe_allow_html=True
    )


    threshold_fig = go.Figure()


    for column_name, label in [
        (
            "recall",
            "Recall"
        ),
        (
            "precision",
            "Precision"
        ),
        (
            "false_warning_rate",
            "False warning rate"
        ),
        (
            "first_alarm_warning_rate",
            "First alarm warning rate"
        )
    ]:

        threshold_fig.add_trace(
            go.Scatter(
                x=threshold_df[
                    "threshold"
                ],

                y=threshold_df[
                    column_name
                ],

                mode="lines+markers",

                name=label
            )
        )


    threshold_fig.add_vline(
        x=0.50,
        line_dash="dash",
        annotation_text="Main threshold"
    )


    threshold_fig.update_layout(
        xaxis_title=
            "Warning threshold",

        yaxis_title=
            "Metric",

        yaxis=dict(
            range=[
                0,
                1
            ],
            tickformat=".0%"
        ),

        height=480
    )


    st.plotly_chart(
        threshold_fig,
        use_container_width=True
    )


    st.caption(
        (
            "Threshold sensitivity is shown only to illustrate "
            "the recall–false-warning trade-off. "
            "The main reported evaluation uses threshold 0.50."
        )
    )


# ============================================================
# PAGE 4. PROCESS EXPLANATION
# ============================================================

elif page == "Process Explanation":

    st.markdown(
        """
<div class="section-title">
Process Explanation
</div>
""",
        unsafe_allow_html=True
    )


    st.write(
        (
            "Process features repeatedly selected during "
            "Leave-One-Lot-Out validation and their SHAP importance."
        )
    )


    top_n = st.slider(
        "Number of process features",
        min_value=5,
        max_value=20,
        value=10
    )


    importance_df = (
        process_importance_df
        .head(
            top_n
        )
        .copy()
    )


    importance_df[
        "display_name"
    ] = (
        importance_df[
            "feature_clean"
        ]
        .apply(
            clean_process_name
        )
    )


    fig = go.Figure(
        go.Bar(
            x=importance_df[
                "mean_abs_shap"
            ],

            y=importance_df[
                "display_name"
            ],

            orientation="h",

            text=[
                f"{x:.3f}"
                for x
                in importance_df[
                    "mean_abs_shap"
                ]
            ],

            textposition="outside"
        )
    )


    fig.update_layout(
        title=
            "Process Feature Importance",

        xaxis_title=
            "Mean |SHAP|",

        yaxis_title=
            "Process feature",

        yaxis=dict(
            autorange="reversed"
        ),

        height=520
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    display_table = (
        importance_df[
            [
                "display_name",
                "selected_folds",
                "mean_selection_rank",
                "mean_abs_shap"
            ]
        ]
        .rename(
            columns={
                "display_name":
                    "Process feature",

                "selected_folds":
                    "Selected LOLO folds",

                "mean_selection_rank":
                    "Mean selection rank",

                "mean_abs_shap":
                    "Mean |SHAP|"
            }
        )
    )


    st.dataframe(
        display_table,
        use_container_width=True,
        hide_index=True
    )


    st.warning(
        (
            "SHAP and feature-selection results describe predictive "
            "associations. They do not establish that changing the "
            "highlighted variable will eliminate process drift."
        )
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
<div class="disclaimer">

<b>Prototype scope:</b>
Historical BOSCH plasma-etching data are replayed in wafer order.
This application is not connected to live fab equipment.

<br>

The AI predicts only the immediately following wafer
(1-step-ahead) and supports engineer review.
It does not automatically modify process recipes.

</div>
""",
    unsafe_allow_html=True
)