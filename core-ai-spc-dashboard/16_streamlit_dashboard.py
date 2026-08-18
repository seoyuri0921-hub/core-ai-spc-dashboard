# ============================================================
# 16_streamlit_dashboard.py
# AI-SPC Process Monitor
# ============================================================

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
    initial_sidebar_state="expanded",
)


# ============================================================
# 1. PATH
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# Local:
# core_drift_ai/
# ├─ app/
# │  └─ 16_streamlit_dashboard.py
# └─ results/
#
# Cloud:
# dashboard_folder/
# ├─ 16_streamlit_dashboard.py
# └─ results/

if (SCRIPT_DIR / "results" / "tables").exists():
    BASE_DIR = SCRIPT_DIR

elif (SCRIPT_DIR.parent / "results" / "tables").exists():
    BASE_DIR = SCRIPT_DIR.parent

else:
    BASE_DIR = SCRIPT_DIR


TABLE_DIR = BASE_DIR / "results" / "tables"

SYSTEM_FILE = TABLE_DIR / "final_ai_spc_system.csv"
METRICS_FILE = TABLE_DIR / "final_evaluation_metrics.csv"
FIRST_ALARM_FILE = TABLE_DIR / "final_first_alarm_performance.csv"
THRESHOLD_FILE = TABLE_DIR / "final_threshold_analysis.csv"
PROCESS_IMPORTANCE_FILE = TABLE_DIR / "shap_process_interpretation.csv"

BASELINE_VALIDATION_FILE = (
    TABLE_DIR / "baseline_sensitivity_summary.csv"
)

MODEL_COMPARISON_FILE = (
    TABLE_DIR / "model_comparison_summary.csv"
)


# ============================================================
# 2. CSS
# ============================================================

st.markdown(
    """
<style>

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

/* ==========================================================
   HEADER
========================================================== */

.main-title {
    font-size: 2.55rem;
    font-weight: 850;
    color: #111827;
    margin-bottom: 0.15rem;
}

.sub-title {
    font-size: 1.05rem;
    color: #6b7280;
    margin-bottom: 1.5rem;
}


/* ==========================================================
   BASIC CARD
========================================================== */

.step-card {
    border: 1px solid #e5e7eb;
    border-radius: 18px;
    padding: 1.35rem 1.45rem;
    min-height: 230px;
    background: white;
    box-shadow: 0 2px 10px rgba(0,0,0,0.04);
}

.step-number {
    font-size: 0.83rem;
    color: #6b7280;
    font-weight: 750;
    margin-bottom: 0.5rem;
}

.step-title {
    font-size: 1.15rem;
    font-weight: 800;
    color: #1f2937;
    margin-bottom: 0.75rem;
}


/* ==========================================================
   EARLY WARNING = ORANGE
========================================================== */

.warning-card {
    border: 3px solid #f59e0b;
    border-radius: 18px;
    padding: 1.35rem 1.45rem;
    min-height: 230px;
    background: #fffaf0;

    box-shadow:
        0 0 0 3px rgba(245,158,11,0.08),
        0 8px 22px rgba(245,158,11,0.15);

    animation: warningPulse 1.8s infinite;
}

@keyframes warningPulse {
    0% {
        box-shadow:
            0 0 0 0 rgba(245,158,11,0.20),
            0 8px 22px rgba(245,158,11,0.12);
    }

    50% {
        box-shadow:
            0 0 0 7px rgba(245,158,11,0.07),
            0 10px 28px rgba(245,158,11,0.20);
    }

    100% {
        box-shadow:
            0 0 0 0 rgba(245,158,11,0.20),
            0 8px 22px rgba(245,158,11,0.12);
    }
}


/* ==========================================================
   SPC ALARM = RED
========================================================== */

.alarm-card {
    border: 3px solid #dc2626;
    border-radius: 18px;
    padding: 1.35rem 1.45rem;
    min-height: 230px;
    background: #fef2f2;

    box-shadow:
        0 0 0 3px rgba(220,38,38,0.10),
        0 8px 22px rgba(220,38,38,0.20);

    animation: alarmPulse 1.5s infinite;
}

@keyframes alarmPulse {
    0% {
        box-shadow:
            0 0 0 0 rgba(220,38,38,0.25),
            0 8px 22px rgba(220,38,38,0.15);
    }

    50% {
        box-shadow:
            0 0 0 7px rgba(220,38,38,0.08),
            0 10px 28px rgba(220,38,38,0.25);
    }

    100% {
        box-shadow:
            0 0 0 0 rgba(220,38,38,0.25),
            0 8px 22px rgba(220,38,38,0.15);
    }
}


/* ==========================================================
   CURRENT STATUS
========================================================== */

.big-status {
    font-size: 2rem;
    font-weight: 900;
    color: #111827;
    margin-top: 0.45rem;
    margin-bottom: 0.65rem;
}

.big-status-alarm {
    font-size: 2.25rem;
    font-weight: 950;
    color: #dc2626 !important;
    margin-top: 0.45rem;
    margin-bottom: 0.65rem;
}


/* ==========================================================
   AI RISK
========================================================== */

.big-risk {
    font-size: 2.5rem;
    font-weight: 900;
    color: #111827;
    line-height: 1;
    margin-top: 0.4rem;
    margin-bottom: 0.7rem;
}

.big-risk-warning {
    font-size: 3rem;
    font-weight: 950;
    color: #ea580c;
    line-height: 1;
    margin-top: 0.4rem;
    margin-bottom: 0.7rem;
}

.big-risk-alarm {
    font-size: 3rem;
    font-weight: 950;
    color: #dc2626;
    line-height: 1;
    margin-top: 0.4rem;
    margin-bottom: 0.7rem;
}


/* ==========================================================
   ENGINEER ACTION
========================================================== */

.engineer-normal {
    font-size: 1.55rem;
    font-weight: 900;
    color: #111827;
    margin-top: 0.55rem;
    margin-bottom: 0.75rem;
    line-height: 1.25;
}

.engineer-warning {
    font-size: 1.55rem;
    font-weight: 950;
    color: #ea580c;
    margin-top: 0.55rem;
    margin-bottom: 0.75rem;
    line-height: 1.25;
}

.engineer-alarm {
    font-size: 1.65rem;
    font-weight: 950;
    color: #dc2626;
    margin-top: 0.55rem;
    margin-bottom: 0.75rem;
    line-height: 1.25;
}


/* ==========================================================
   DESCRIPTION
========================================================== */

.small-description {
    color: #4b5563;
    font-size: 0.94rem;
    line-height: 1.5;
}

.warning-description {
    color: #9a3412;
    font-size: 0.78rem;
    font-weight: 600;
    line-height: 1.45;
}

.alarm-description {
    color: #991b1b;
    font-size: 0.88rem;
    font-weight: 650;
    line-height: 1.45;
}


/* ==========================================================
   BADGES
========================================================== */

.badge-normal {
    display: inline-block;
    padding: 0.45rem 0.9rem;
    border-radius: 999px;
    background: #dcfce7;
    color: #166534;
    font-weight: 800;
}

.badge-warning {
    display: inline-block;
    padding: 0.5rem 1rem;
    border-radius: 999px;
    background: #ffedd5;
    color: #c2410c;
    border: 1px solid #f59e0b;
    font-weight: 900;
    font-size: 1.05rem;
}

.badge-alarm {
    display: inline-block;
    padding: 0.5rem 1rem;
    border-radius: 999px;
    background: #fee2e2;
    color: #dc2626;
    border: 1px solid #ef4444;
    font-weight: 950;
    font-size: 1.05rem;
}


/* ==========================================================
   BANNERS
========================================================== */

.warning-banner {
    border: 2px solid #f59e0b;
    background: #fff7ed;
    color: #9a3412;
    padding: 1rem 1.2rem;
    border-radius: 14px;
    font-size: 1.05rem;
    font-weight: 750;
    margin-top: 0.8rem;
    margin-bottom: 0.7rem;
}

.alarm-banner {
    border: 2px solid #ef4444;
    background: #fef2f2;
    color: #b91c1c;
    padding: 1rem 1.2rem;
    border-radius: 14px;
    font-size: 1.05rem;
    font-weight: 800;
    margin-top: 0.8rem;
    margin-bottom: 0.7rem;
}


/* ==========================================================
   SECTION
========================================================== */

.section-title {
    font-size: 1.45rem;
    font-weight: 850;
    color: #111827;
    margin-top: 1.1rem;
    margin-bottom: 0.75rem;
}


/* ==========================================================
   CONTRIBUTOR
========================================================== */

.contributor-card {
    border: 1px solid #e5e7eb;
    border-radius: 13px;
    padding: 1rem;
    background: #f9fafb;
    min-height: 145px;
}

.contributor-rank {
    color: #6b7280;
    font-size: 0.82rem;
    font-weight: 800;
}

.contributor-name {
    margin-top: 0.35rem;
    margin-bottom: 0.55rem;
    font-weight: 800;
    color: #1f2937;
}


/* ==========================================================
   FOOTER
========================================================== */

.disclaimer {
    color: #6b7280;
    font-size: 0.84rem;
    line-height: 1.5;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# 3. FILE CHECK
# ============================================================

required_files = [
    SYSTEM_FILE,
    METRICS_FILE,
    FIRST_ALARM_FILE,
    THRESHOLD_FILE,
    PROCESS_IMPORTANCE_FILE,
]

missing_files = [
    file
    for file in required_files
    if not file.exists()
]

if missing_files:

    st.error(
        "Dashboard data files could not be found."
    )

    st.write(
        "Detected base directory:",
        str(BASE_DIR),
    )

    st.write(
        "Missing files:"
    )

    for file in missing_files:
        st.code(str(file))

    st.stop()


# ============================================================
# 4. DATA LOAD
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


    if BASELINE_VALIDATION_FILE.exists():

        baseline_validation_df = pd.read_csv(
            BASELINE_VALIDATION_FILE
        )

    else:

        baseline_validation_df = pd.DataFrame()


    if MODEL_COMPARISON_FILE.exists():

        model_comparison_df = pd.read_csv(
            MODEL_COMPARISON_FILE
        )

    else:

        model_comparison_df = pd.DataFrame()


    bool_columns = [
        "ai_warning",
        "successful_first_alarm_one_step_warning",
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
                        "false": False,
                    }
                )
            )


    return (
        system_df,
        metrics_df,
        first_alarm_df,
        threshold_df,
        process_importance_df,
        baseline_validation_df,
        model_comparison_df,
    )


(
    system_df,
    metrics_df,
    first_alarm_df,
    threshold_df,
    process_importance_df,
    baseline_validation_df,
    model_comparison_df,
) = load_data()


# ============================================================
# 5. HELPERS
# ============================================================

def get_metric(
    metric_name,
    default=np.nan,
):

    row = metrics_df[
        metrics_df["metric"]
        == metric_name
    ]

    if len(row) == 0:
        return default

    return float(
        row.iloc[0]["value"]
    )


def clean_process_name(name):

    if pd.isna(name):
        return "-"

    clean = str(name)

    clean = clean.replace(
        "current_",
        "",
    )

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
            "Chamber Pressure",
    }

    for old, new in replacements.items():

        clean = clean.replace(
            old,
            new,
        )

    clean = clean.replace(
        "_mean",
        " (mean)",
    )

    clean = clean.replace(
        "_std",
        " (variation)",
    )

    clean = clean.replace(
        "_min",
        " (minimum)",
    )

    clean = clean.replace(
        "_max",
        " (maximum)",
    )

    return clean


def simple_status_badge(status):

    if status == "NORMAL":

        return (
            '<span class="badge-normal">'
            '● 정상'
            '</span>'
        )

    if status == "EARLY WARNING":

        return (
            '<span class="badge-warning">'
            '⚠ 사전 경고'
            '</span>'
        )

    if status == "SPC ALARM":

        return (
            '<span class="badge-alarm">'
            '🚨 SPC ALARM'
            '</span>'
        )

    return str(status)


def result_text(result):

    if result == "TRUE EARLY WARNING":

        return (
            "✅ AI가 실제 이상을 사전에 경고했습니다."
        )

    if result == "FALSE EARLY WARNING":

        return (
            "⚠️ AI가 경고했지만 실제 다음 wafer는 정상이었습니다."
        )

    if result == "MISSED EARLY WARNING":

        return (
            "❌ AI가 위험을 놓쳤고 실제 다음 wafer에서 이상이 발생했습니다."
        )

    if result == "CORRECT NORMAL":

        return (
            "✅ AI가 정상으로 판단했고 실제 다음 wafer도 정상이었습니다."
        )

    return str(result)


# ============================================================
# 6. SIDEBAR
# ============================================================

st.sidebar.markdown(
    "## ⚙️ AI-SPC Monitor"
)


page = st.sidebar.radio(
    "화면 선택",
    [
        "📡 Live Monitor",
        "📈 Lot Timeline",
        "🔬 Model Validation",
    ],
)


st.sidebar.markdown("---")


st.sidebar.markdown(
    """
### 연구 흐름

**현재 wafer 확인**

↓

**AI가 다음 wafer 예측**

↓

**위험 시 사전 경고**

↓

**엔지니어 최종 판단**
"""
)


st.sidebar.caption(
    "Prediction horizon: exactly 1 wafer ahead"
)


st.sidebar.caption(
    "AI는 공정 recipe를 자동 변경하지 않습니다."
)


# ============================================================
# 7. HEADER
# ============================================================

st.markdown(
    """
<div class="main-title">
AI–SPC Process Monitor
</div>

<div class="sub-title">
현재 wafer의 공정 정보를 이용해 바로 다음 wafer의 SPC 이상 위험을 예측하는 사전경고 시스템
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# PAGE 1. LIVE MONITOR
# ============================================================

if page == "📡 Live Monitor":


    # ========================================================
    # SELECTOR
    # ========================================================

    select1, select2, select3 = st.columns(
        [
            1,
            1,
            2,
        ]
    )


    lot_list = sorted(
        system_df[
            "lot_number"
        ].unique()
    )


    with select1:

        selected_lot = st.selectbox(
            "Lot 선택",
            lot_list,
        )


    lot_df = (
        system_df[
            system_df["lot_number"]
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


    with select2:

        selected_current_wafer = st.selectbox(
            "현재 Wafer",
            wafer_list,
        )


    selected_row = (
        lot_df[
            lot_df["current_wafer"]
            == selected_current_wafer
        ]
        .iloc[0]
    )


    next_wafer = int(
        selected_row["next_wafer"]
    )


    with select3:

        st.markdown(
            f"""
<div style="
padding-top:1.8rem;
font-size:1.2rem;
font-weight:800;
color:#374151;
">
Lot {int(selected_lot)}
&nbsp;&nbsp;|&nbsp;&nbsp;
현재 W{int(selected_current_wafer)}
&nbsp;&nbsp;→&nbsp;&nbsp;
다음 W{next_wafer} 예측
</div>
""",
            unsafe_allow_html=True,
        )


    st.markdown("---")


    # ========================================================
    # CURRENT DATA
    # ========================================================

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


    risk = float(
        selected_row[
            "predicted_risk"
        ]
    )


    # ========================================================
    # STYLE SELECTION
    # ========================================================

    if current_spc_status == "SPC ALARM":

        card1_class = "alarm-card"

        current_status_class = (
            "big-status-alarm"
        )

        current_status_display = (
            "🚨 ALARM"
        )

    else:

        card1_class = "step-card"

        current_status_class = (
            "big-status"
        )

        current_status_display = (
            simple_status_badge(
                current_spc_status
            )
        )


    if system_status == "EARLY WARNING":

        card2_class = (
            "warning-card"
        )

        risk_class = (
            "big-risk-warning"
        )

        step2_description_class = (
            "warning-description"
        )


    elif system_status == "SPC ALARM":

        card2_class = (
            "alarm-card"
        )

        risk_class = (
            "big-risk-alarm"
        )

        step2_description_class = (
            "alarm-description"
        )


    else:

        card2_class = (
            "step-card"
        )

        risk_class = (
            "big-risk"
        )

        step2_description_class = (
            "small-description"
        )


    if system_status == "EARLY WARNING":

        card3_class = (
            "warning-card"
        )

        engineer_class = (
            "engineer-warning"
        )

        description_class = (
            "warning-description"
        )


    elif system_status == "SPC ALARM":

        card3_class = (
            "alarm-card"
        )

        engineer_class = (
            "engineer-alarm"
        )

        description_class = (
            "alarm-description"
        )


    else:

        card3_class = (
            "step-card"
        )

        engineer_class = (
            "engineer-normal"
        )

        description_class = (
            "small-description"
        )


    # ========================================================
    # MAIN 3 STEP FLOW
    # ========================================================

    card1, arrow1, card2, arrow2, card3 = st.columns(
        [
            4,
            0.55,
            4,
            0.55,
            4,
        ]
    )


    # ========================================================
    # STEP 1
    # ========================================================

    with card1:

        if current_spc_status == "SPC ALARM":

            current_description = (
                "현재 wafer가 SPC 관리한계를 벗어났습니다. "
                "공정 상태 확인이 필요합니다."
            )

        else:

            current_description = (
                "현재 wafer의 실제 측정 결과를 "
                "기존 SPC 기준으로 확인합니다."
            )


        st.markdown(
            f"""
<div class="{card1_class}">

<div class="step-number">
STEP 1
</div>

<div class="step-title">
현재 공정 상태
</div>

<div class="{current_status_class}">
{current_status_display}
</div>

<div class="small-description">

현재 W{int(selected_current_wafer)}의 평균 식각 깊이:

<br><br>

<b>
{selected_row['current_mean_si_etch']:.3f}
</b>

<br><br>

{current_description}

</div>

</div>
""",
            unsafe_allow_html=True,
        )


    # ========================================================
    # ARROW 1
    # ========================================================

    with arrow1:

        st.markdown(
            """
<div style="
font-size:2rem;
text-align:center;
padding-top:5rem;
color:#9ca3af;
">
→
</div>
""",
            unsafe_allow_html=True,
        )


    # ========================================================
    # STEP 2
    # ========================================================

    with card2:

        if system_status == "EARLY WARNING":

            prediction_text = (
                f"다음 W{next_wafer}에서 "
                "SPC 이상 가능성이 높다고 판단했습니다."
            )

        elif system_status == "SPC ALARM":

            prediction_text = (
                "현재 SPC가 이미 이상 상태를 감지했습니다."
            )

        else:

            prediction_text = (
                f"다음 W{next_wafer}의 "
                "SPC 이상 위험이 낮다고 판단했습니다."
            )


        st.markdown(
            f"""
<div class="{card2_class}">

<div class="step-number">
STEP 2
</div>

<div class="step-title">
AI 다음 Wafer 예측
</div>

<div class="{risk_class}">
{risk:.0%}
</div>

<div>
{simple_status_badge(system_status)}
</div>

<br>

<div class="{step2_description_class}">
{prediction_text}
</div>

</div>
""",
            unsafe_allow_html=True,
        )


    # ========================================================
    # ARROW 2
    # ========================================================

    with arrow2:

        st.markdown(
            """
<div style="
font-size:2rem;
text-align:center;
padding-top:5rem;
color:#9ca3af;
">
→
</div>
""",
            unsafe_allow_html=True,
        )


    # ========================================================
    # STEP 3
    # ========================================================

    with card3:

        if system_status == "EARLY WARNING":

            action_title = (
                '<span style="white-space:nowrap;">'
                '⚠ 엔지니어'
                '</span>'
                '<br>'
                '<span style="white-space:nowrap;">'
                '확인 필요'
                '</span>'
            )

            action_description = (
                "AI가 다음 wafer의 위험 신호를 감지했습니다. "
                "주요 공정 센서를 확인한 뒤 공정 조정 여부를 "
                "엔지니어가 결정합니다."
            )

        elif system_status == "SPC ALARM":

            action_title = (
                '<span style="white-space:nowrap;">'
                '🚨 즉시 공정'
                '</span>'
                '<br>'
                '<span style="white-space:nowrap;">'
                '확인 필요'
                '</span>'
            )

            action_description = (
                "기존 SPC가 이미 이상을 감지했습니다. "
                "엔지니어의 공정 점검이 필요합니다."
            )

        else:

            action_title = (
                "✅ 계속 모니터링"
            )

            action_description = (
                "현재 AI 사전경고는 없습니다. "
                "다음 wafer까지 공정을 계속 모니터링합니다."
            )


        st.markdown(
            f"""
<div class="{card3_class}">

<div class="step-number">
STEP 3
</div>

<div class="step-title">
최종 판단
</div>

<div class="{engineer_class}">
{action_title}
</div>

<div class="{description_class}">
{action_description}
</div>

</div>
""",
            unsafe_allow_html=True,
        )


    # ========================================================
    # SUMMARY BANNER
    # ========================================================

    if (
        current_spc_status == "NORMAL"
        and system_status == "EARLY WARNING"
    ):

        st.markdown(
            f"""
<div class="warning-banner">

⚠ <b>AI EARLY WARNING</b>
&nbsp;&nbsp;|&nbsp;&nbsp;

현재 W{int(selected_current_wafer)}는 SPC 기준 정상

&nbsp;&nbsp;→&nbsp;&nbsp;

다음 W{next_wafer} 이상 위험

<span style="
font-size:1.35rem;
font-weight:950;
color:#ea580c;
">
{risk:.0%}
</span>

&nbsp;&nbsp;→&nbsp;&nbsp;

<b>엔지니어 확인 필요</b>

</div>
""",
            unsafe_allow_html=True,
        )


    elif current_spc_status == "SPC ALARM":

        st.markdown(
            f"""
<div class="alarm-banner">

🚨 <b>SPC ALARM</b>

&nbsp;&nbsp;|&nbsp;&nbsp;

현재 W{int(selected_current_wafer)}가
SPC 관리한계를 벗어났습니다.

&nbsp;&nbsp;→&nbsp;&nbsp;

<b>즉시 공정 상태 확인 필요</b>

</div>
""",
            unsafe_allow_html=True,
        )


    else:

        st.success(
            (
                f"현재 W{int(selected_current_wafer)}는 정상이며, "
                f"AI 역시 다음 W{next_wafer}의 위험을 "
                f"낮게 평가했습니다."
            )
        )


    # ========================================================
    # DETAILS
    # ========================================================

    with st.expander(
        "🔎 상세 분석 보기"
    ):

        st.markdown(
            "### 기존 SPC 기준"
        )


        spc1, spc2, spc3, spc4 = st.columns(
            4
        )


        spc1.metric(
            "현재 식각 깊이",
            f"{selected_row['current_mean_si_etch']:.3f}",
        )


        spc2.metric(
            "LCL",
            f"{selected_row['LCL']:.3f}",
        )


        spc3.metric(
            "Center Line",
            f"{selected_row['CL']:.3f}",
        )


        spc4.metric(
            "UCL",
            f"{selected_row['UCL']:.3f}",
        )


        st.markdown(
            "### AI 판단에 기여한 주요 공정 신호"
        )


        contributor_cols = st.columns(
            3
        )


        for rank in range(
            1,
            4,
        ):

            process_name = selected_row[
                f"top{rank}_process"
            ]

            process_value = selected_row[
                f"top{rank}_value"
            ]

            process_shap_value = selected_row[
                f"top{rank}_shap"
            ]


            with contributor_cols[
                rank - 1
            ]:

                if pd.isna(
                    process_name
                ):

                    st.info(
                        "표시할 공정 변수가 없습니다."
                    )

                else:

                    st.markdown(
                        f"""
<div class="contributor-card">

<div class="contributor-rank">
#{rank} 주요 공정 신호
</div>

<div class="contributor-name">
{clean_process_name(process_name)}
</div>

센서 요약값:
<b>{process_value:.4f}</b>

<br>

AI 기여도:
<b>{process_shap_value:+.3f}</b>

</div>
""",
                        unsafe_allow_html=True,
                    )


        st.caption(
            (
                "SHAP은 AI가 어떤 공정 신호를 중요하게 사용했는지 "
                "설명합니다. 해당 변수가 실제 이상 원인이라는 뜻은 아닙니다."
            )
        )


    # ========================================================
    # RETROSPECTIVE VALIDATION
    # ========================================================

    with st.expander(
        "🧪 실제 다음 Wafer 결과 확인"
    ):

        st.caption(
            (
                "다음 wafer 결과는 실제 예측 시점에는 알 수 없습니다. "
                "연구에서 AI 예측이 맞았는지 확인하기 위한 사후 검증입니다."
            )
        )


        actual_status = str(
            selected_row[
                "next_actual_status"
            ]
        )


        warning_result = str(
            selected_row[
                "warning_result"
            ]
        )


        result_col1, result_col2 = st.columns(
            2
        )


        with result_col1:

            st.metric(
                f"실제 W{next_wafer} 식각 깊이",
                f"{selected_row['next_actual_mean_si_etch']:.3f}",
            )


            st.write(
                "실제 SPC 상태:"
            )


            if actual_status == "SPC VIOLATION":

                st.error(
                    "🚨 SPC 이상 발생"
                )

            else:

                st.success(
                    "🟢 정상"
                )


        with result_col2:

            st.write(
                "AI 예측 평가:"
            )


            result_message = result_text(
                warning_result
            )


            if warning_result == "TRUE EARLY WARNING":

                st.success(
                    result_message
                )

            elif warning_result == "FALSE EARLY WARNING":

                st.warning(
                    result_message
                )

            elif warning_result == "MISSED EARLY WARNING":

                st.error(
                    result_message
                )

            else:

                st.success(
                    result_message
                )


        if bool(
            selected_row[
                "successful_first_alarm_one_step_warning"
            ]
        ):

            st.success(
                (
                    f"🎯 AI가 W{int(selected_current_wafer)} 시점에서 "
                    f"W{next_wafer} 위험을 경고했고, "
                    f"실제로 W{next_wafer}에서 "
                    f"최초 SPC 이상이 발생했습니다."
                )
            )


# ============================================================
# PAGE 2. LOT TIMELINE
# ============================================================

elif page == "📈 Lot Timeline":

    st.markdown(
        """
<div class="section-title">
Lot 전체 흐름
</div>
""",
        unsafe_allow_html=True,
    )


    st.write(
        (
            "한 Lot 안에서 AI 위험도가 어떻게 변하고, "
            "언제 실제 SPC 이상이 발생했는지 확인합니다."
        )
    )


    lot_list = sorted(
        system_df[
            "lot_number"
        ].unique()
    )


    selected_lot = st.selectbox(
        "Lot 선택",
        lot_list,
        key="timeline_lot",
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
    # RISK CHART
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
            mode="lines+markers",
            name="AI 위험도",
            hovertemplate=(
                "예측 대상 W%{x}"
                "<br>"
                "위험도 %{y:.1%}"
                "<extra></extra>"
            ),
        )
    )


    fig.add_hline(
        y=0.50,
        line_dash="dash",
        annotation_text="사전경고 기준 50%",
    )


    warning_rows = lot_df[
        lot_df[
            "ai_warning"
        ]
        == True
    ]


    if len(warning_rows) > 0:

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
                    size=16,
                    symbol="triangle-up",
                ),
                name="AI 사전경고",
            )
        )


    first_alarm_rows = lot_df[
        lot_df[
            "next_first_spc_alarm"
        ]
        == 1
    ]


    if len(first_alarm_rows) > 0:

        first_alarm_wafer = int(
            first_alarm_rows[
                "next_wafer"
            ].iloc[0]
        )


        fig.add_vline(
            x=first_alarm_wafer,
            line_dash="dot",
            annotation_text=(
                f"최초 SPC 이상 W"
                f"{first_alarm_wafer}"
            ),
        )


    fig.update_layout(
        title=(
            f"Lot {int(selected_lot)} "
            "— 다음 Wafer AI 위험도"
        ),
        xaxis_title="예측 대상 Wafer",
        yaxis_title="AI 위험도",
        yaxis=dict(
            range=[
                0,
                1,
            ],
            tickformat=".0%",
        ),
        height=480,
        hovermode="x unified",
    )


    st.plotly_chart(
        fig,
        use_container_width=True,
    )


    # ========================================================
    # EVENT SUMMARY
    # ========================================================

    if len(first_alarm_rows) > 0:

        first_row = (
            first_alarm_rows
            .iloc[0]
        )


        target_wafer = int(
            first_row[
                "next_wafer"
            ]
        )


        current_wafer = int(
            first_row[
                "current_wafer"
            ]
        )


        risk_value = float(
            first_row[
                "predicted_risk"
            ]
        )


        warned = bool(
            first_row[
                "successful_first_alarm_one_step_warning"
            ]
        )


        if warned:

            st.success(
                (
                    f"🎯 Lot {int(selected_lot)}: "
                    f"W{current_wafer} 처리 후 AI가 "
                    f"W{target_wafer} 위험을 "
                    f"{risk_value:.0%}로 경고했고, "
                    f"실제로 W{target_wafer}에서 "
                    f"최초 SPC 이상이 발생했습니다."
                )
            )

        else:

            st.warning(
                (
                    f"Lot {int(selected_lot)}: "
                    f"W{target_wafer}에서 최초 SPC 이상이 발생했지만, "
                    f"직전 W{current_wafer}의 AI 위험도는 "
                    f"{risk_value:.0%}로 사전경고 기준에 미달했습니다."
                )
            )

    else:

        st.info(
            (
                f"Lot {int(selected_lot)}에서는 "
                "관측 가능한 wafer 구간 내 "
                "최초 SPC 이상이 확인되지 않았습니다."
            )
        )


    # ========================================================
    # DETAIL TABLE
    # ========================================================

    with st.expander(
        "📋 상세 Lot 데이터 보기"
    ):

        display_df = (
            lot_df[
                [
                    "current_wafer",
                    "next_wafer",
                    "current_spc_status",
                    "predicted_risk",
                    "system_status",
                    "next_actual_status",
                    "warning_result",
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
                    "현재 Wafer",

                "next_wafer":
                    "예측 대상",

                "current_spc_status":
                    "현재 SPC",

                "predicted_risk":
                    "AI 위험도",

                "system_status":
                    "AI 판단",

                "next_actual_status":
                    "실제 다음 결과",

                "warning_result":
                    "평가",
            }
        )


        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# PAGE 3. MODEL VALIDATION
# ============================================================

elif page == "🔬 Model Validation":

    st.markdown(
        """
<div class="section-title">
모델 검증 결과
</div>
""",
        unsafe_allow_html=True,
    )


    st.write(
        (
            "AI가 얼마나 잘 맞혔는지와 "
            "SPC baseline 및 AI 모델 선택의 타당성을 확인합니다."
        )
    )


    # ========================================================
    # 1. CURRENT PERFORMANCE
    # ========================================================

    st.markdown(
        "### 1. 현재 AI Early-Warning 성능"
    )


    precision = get_metric(
        "precision"
    )

    recall = get_metric(
        "recall"
    )

    specificity = get_metric(
        "specificity"
    )

    roc_auc = get_metric(
        "roc_auc"
    )

    false_warning_rate = get_metric(
        "false_warning_rate"
    )

    first_warning_rate = get_metric(
        "first_spc_alarm_warning_rate"
    )


    metrics_cols = st.columns(
        6
    )


    metrics_values = [
        (
            "이상 탐지율",
            recall,
            "%",
        ),
        (
            "경고 정확도",
            precision,
            "%",
        ),
        (
            "정상 판별률",
            specificity,
            "%",
        ),
        (
            "ROC-AUC",
            roc_auc,
            "",
        ),
        (
            "오경고율",
            false_warning_rate,
            "%",
        ),
        (
            "최초 이상 사전경고",
            first_warning_rate,
            "%",
        ),
    ]


    for column, (
        label,
        value,
        kind,
    ) in zip(
        metrics_cols,
        metrics_values,
    ):

        if kind == "%":

            display_value = (
                f"{value:.1%}"
            )

        else:

            display_value = (
                f"{value:.3f}"
            )


        with column:

            st.metric(
                label,
                display_value,
            )


    # ========================================================
    # 2. BASELINE VALIDATION
    # ========================================================

    st.markdown("---")


    st.markdown(
        "### 2. 왜 W1~W3를 SPC baseline으로 사용했나?"
    )


    if len(
        baseline_validation_df
    ) > 0:

        baseline_show = (
            baseline_validation_df[
                [
                    "baseline_definition",
                    "pooled_sigma",
                    "three_sigma_half_width",
                    "lots_with_spc_alarm",
                ]
            ]
            .copy()
        )


        baseline_show = baseline_show.rename(
            columns={
                "baseline_definition":
                    "Baseline",

                "pooled_sigma":
                    "Pooled σ",

                "three_sigma_half_width":
                    "±3σ 폭",

                "lots_with_spc_alarm":
                    "SPC alarm Lot",
            }
        )


        st.dataframe(
            baseline_show,
            use_container_width=True,
            hide_index=True,
        )


        st.success(
            (
                "W1~W3에서 pooled σ와 lot 간 baseline 변동성이 가장 작았습니다. "
                "반면 W4를 포함하면 10개 lot 중 9개에서 baseline center가 "
                "drift 방향으로 하향 이동했습니다."
            )
        )

    else:

        st.info(
            (
                "baseline_sensitivity_summary.csv를 "
                "results/tables 폴더에 추가하면 결과가 표시됩니다."
            )
        )


    # ========================================================
    # 3. MODEL COMPARISON
    # ========================================================

    st.markdown("---")


    st.markdown(
        "### 3. AI 모델 비교"
    )


    if len(
        model_comparison_df
    ) > 0:

        candidate_columns = [
            "model",
            "roc_auc",
            "recall",
            "precision",
            "specificity",
            "f1",
            "false_warning_rate",
            "warning_rate",
        ]


        available_columns = [
            col
            for col in candidate_columns
            if col in model_comparison_df.columns
        ]


        model_show = (
            model_comparison_df[
                available_columns
            ]
            .copy()
        )


        rename_dict = {
            "model":
                "Model",

            "roc_auc":
                "ROC-AUC",

            "recall":
                "Recall",

            "precision":
                "Precision",

            "specificity":
                "Specificity",

            "f1":
                "F1",

            "false_warning_rate":
                "False warning",

            "warning_rate":
                "First alarm warning",
        }


        model_show = model_show.rename(
            columns=rename_dict
        )


        st.dataframe(
            model_show,
            use_container_width=True,
            hide_index=True,
        )


        st.info(
            (
                "현재 비교에서는 SVM이 전반적 판별력과 낮은 오경고율에서 "
                "우수했고, XGBoost는 최초 SPC 이상 사전경고율에서 가장 높았습니다."
            )
        )

    else:

        st.info(
            (
                "model_comparison_summary.csv를 "
                "results/tables 폴더에 추가하면 모델 비교 결과가 표시됩니다."
            )
        )


    # ========================================================
    # 4. THRESHOLD
    # ========================================================

    st.markdown("---")


    with st.expander(
        "Threshold별 성능 비교 보기"
    ):

        threshold_fig = go.Figure()


        columns_to_plot = [
            (
                "recall",
                "Recall",
            ),
            (
                "false_warning_rate",
                "False warning",
            ),
            (
                "first_alarm_warning_rate",
                "First alarm warning",
            ),
        ]


        for column_name, label in columns_to_plot:

            if (
                column_name
                not in threshold_df.columns
            ):
                continue


            threshold_fig.add_trace(
                go.Scatter(
                    x=threshold_df[
                        "threshold"
                    ],
                    y=threshold_df[
                        column_name
                    ],
                    mode="lines+markers",
                    name=label,
                )
            )


        threshold_fig.add_vline(
            x=0.50,
            line_dash="dash",
            annotation_text="현재 기준 0.50",
        )


        threshold_fig.update_layout(
            xaxis_title="Warning threshold",
            yaxis_title="Performance",
            yaxis=dict(
                range=[
                    0,
                    1,
                ],
                tickformat=".0%",
            ),
            height=450,
        )


        st.plotly_chart(
            threshold_fig,
            use_container_width=True,
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")


st.markdown(
    """
<div class="disclaimer">

<b>연구용 Prototype</b><br>

본 시스템은 historical BOSCH plasma-etching 데이터를
wafer 순서대로 재생한 연구용 prototype입니다.

AI는 현재 wafer까지 확보 가능한 정보를 이용해
바로 다음 wafer 한 장의 SPC 이상 위험만 예측합니다.

AI가 recipe를 자동 변경하지 않으며,
최종 공정 확인 및 조정 여부는 엔지니어가 판단합니다.

</div>
""",
    unsafe_allow_html=True,
)
