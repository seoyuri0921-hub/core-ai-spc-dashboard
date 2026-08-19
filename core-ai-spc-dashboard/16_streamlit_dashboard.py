# ============================================================
# 16_streamlit_dashboard.py
# AI-SPC Process Monitor
# Korean / English switchable version
#
# FINAL CONSISTENT VERSION
#
# Final model:
# XGBoost
# History + Conditioning + Top10 Process Features
#
# Validation:
# Leave-One-Lot-Out (LOLO)
#
# Early-warning evaluation:
# pre_alarm_eligible == 1
#
# Final operating threshold:
# 0.60
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

if (SCRIPT_DIR / "results" / "tables").exists():
    BASE_DIR = SCRIPT_DIR

elif (SCRIPT_DIR.parent / "results" / "tables").exists():
    BASE_DIR = SCRIPT_DIR.parent

else:
    BASE_DIR = SCRIPT_DIR


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

BASELINE_VALIDATION_FILE = (
    TABLE_DIR
    / "baseline_sensitivity_summary.csv"
)

MODEL_COMPARISON_FILE = (
    TABLE_DIR
    / "model_comparison_summary.csv"
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

.post-spc-banner {
    border: 1px solid #9ca3af;
    background: #f9fafb;
    color: #4b5563;
    padding: 0.9rem 1.1rem;
    border-radius: 12px;
    font-size: 0.92rem;
    margin-top: 0.8rem;
    margin-bottom: 0.7rem;
}

.section-title {
    font-size: 1.45rem;
    font-weight: 850;
    color: #111827;
    margin-top: 1.1rem;
    margin-bottom: 0.75rem;
}

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
        str(
            BASE_DIR
        )
    )

    st.write(
        "Missing files:"
    )

    for file in missing_files:
        st.code(
            str(
                file
            )
        )

    st.stop()


# ============================================================
# 4. DATA LOAD
# ============================================================

def file_signature(
    file_path
):

    if not file_path.exists():
        return None

    return (
        file_path.stat().st_mtime_ns
    )


@st.cache_data
def load_data(
    system_sig,
    metrics_sig,
    first_alarm_sig,
    threshold_sig,
    process_sig,
    baseline_sig,
    comparison_sig,
):

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

            system_df[
                column
            ] = (
                system_df[
                    column
                ]
                .astype(str)
                .str.lower()
                .map(
                    {
                        "true": True,
                        "false": False
                    }
                )
            )


    integer_columns = [
        "current_spc_alarm",
        "next_spc_violation",
        "next_first_spc_alarm",
        "pre_alarm_eligible",
    ]


    for column in integer_columns:

        if column in system_df.columns:

            system_df[
                column
            ] = (
                system_df[
                    column
                ]
                .astype(int)
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
) = load_data(

    file_signature(
        SYSTEM_FILE
    ),

    file_signature(
        METRICS_FILE
    ),

    file_signature(
        FIRST_ALARM_FILE
    ),

    file_signature(
        THRESHOLD_FILE
    ),

    file_signature(
        PROCESS_IMPORTANCE_FILE
    ),

    file_signature(
        BASELINE_VALIDATION_FILE
    ),

    file_signature(
        MODEL_COMPARISON_FILE
    ),
)


# ============================================================
# 5. LANGUAGE / HELPERS
# ============================================================

language = st.sidebar.selectbox(
    "🌐 Language / 언어",
    [
        "한국어",
        "English"
    ],
    index=0,
)


LANG = (
    "ko"
    if language == "한국어"
    else "en"
)


def tr(
    ko,
    en
):

    return (
        ko
        if LANG == "ko"
        else en
    )


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


    if len(
        row
    ) == 0:

        return default


    return float(
        row
        .iloc[0][
            "value"
        ]
    )


def clean_process_name(
    name
):

    if pd.isna(
        name
    ):
        return "-"


    clean = (
        str(
            name
        )
        .replace(
            "current_",
            ""
        )
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


    for old, new in (
        replacements.items()
    ):

        clean = clean.replace(
            old,
            new
        )


    if LANG == "ko":

        clean = clean.replace(
            "_mean",
            " (평균)"
        )

        clean = clean.replace(
            "_std",
            " (변동)"
        )

        clean = clean.replace(
            "_min",
            " (최솟값)"
        )

        clean = clean.replace(
            "_max",
            " (최댓값)"
        )


    else:

        clean = clean.replace(
            "_mean",
            " (mean)"
        )

        clean = clean.replace(
            "_std",
            " (variation)"
        )

        clean = clean.replace(
            "_min",
            " (minimum)"
        )

        clean = clean.replace(
            "_max",
            " (maximum)"
        )


    return clean


def simple_status_badge(
    status
):

    if status == "NORMAL":

        label = tr(
            "● 정상",
            "● NORMAL"
        )

        return (
            f'<span class="badge-normal">'
            f'{label}'
            f'</span>'
        )


    if status == "EARLY WARNING":

        label = tr(
            "⚠ 사전 경고",
            "⚠ EARLY WARNING"
        )

        return (
            f'<span class="badge-warning">'
            f'{label}'
            f'</span>'
        )


    if status == "SPC ALARM":

        label = tr(
            "🚨 SPC 이상",
            "🚨 SPC ALARM"
        )

        return (
            f'<span class="badge-alarm">'
            f'{label}'
            f'</span>'
        )


    return str(
        status
    )


def result_text(
    result
):

    messages = {

        "TRUE EARLY WARNING":
            tr(
                "✅ AI가 실제 이상을 사전에 경고했습니다.",
                "✅ The AI successfully warned before the actual anomaly.",
            ),

        "FALSE EARLY WARNING":
            tr(
                "⚠️ AI가 경고했지만 실제 다음 wafer는 정상이었습니다.",
                "⚠️ The AI issued a warning, but the next wafer was actually normal.",
            ),

        "MISSED EARLY WARNING":
            tr(
                "❌ AI가 위험을 놓쳤고 실제 다음 wafer에서 이상이 발생했습니다.",
                "❌ The AI missed the risk, and an anomaly occurred on the next wafer.",
            ),

        "CORRECT NORMAL":
            tr(
                "✅ AI가 정상으로 판단했고 실제 다음 wafer도 정상이었습니다.",
                "✅ The AI predicted normal, and the next wafer was actually normal.",
            ),

        "POST-FIRST-SPC":
            tr(
                "ℹ️ 최초 SPC 이상 발생 이후 구간으로, 사전경고 성능 평가에서는 제외됩니다.",
                "ℹ️ This sample occurs after the first SPC anomaly and is excluded from early-warning performance evaluation.",
            ),
    }


    return messages.get(
        result,
        str(
            result
        )
    )


# ============================================================
# 6. SIDEBAR
# ============================================================

st.sidebar.markdown(
    "## ⚙️ AI-SPC Monitor"
)


PAGE_LABELS = {

    "live":
        tr(
            "📡 실시간 모니터",
            "📡 Live Monitor"
        ),

    "timeline":
        tr(
            "📈 Lot 타임라인",
            "📈 Lot Timeline"
        ),

    "validation":
        tr(
            "🔬 모델 검증",
            "🔬 Model Validation"
        ),
}


page = st.sidebar.radio(
    tr(
        "화면 선택",
        "Select page"
    ),

    [
        "live",
        "timeline",
        "validation"
    ],

    format_func=lambda x:
        PAGE_LABELS[
            x
        ],
)


st.sidebar.markdown(
    "---"
)


st.sidebar.markdown(
    tr(
        """
### 연구 흐름

**현재 wafer 확인**

↓

**AI가 다음 wafer 예측**

↓

**위험 시 사전 경고**

↓

**엔지니어 최종 판단**
""",
        """
### Workflow

**Check current wafer**

↓

**AI predicts next wafer**

↓

**Early warning if risky**

↓

**Engineer makes final decision**
""",
    )
)


st.sidebar.caption(
    tr(
        "예측 범위: 정확히 다음 wafer 1장",
        "Prediction horizon: exactly 1 wafer ahead",
    )
)


st.sidebar.caption(
    tr(
        "최종 AI 모델: XGBoost + History/Conditioning + Top10 Process Features",
        "Final AI model: XGBoost + History/Conditioning + Top10 Process Features",
    )
)


st.sidebar.caption(
    tr(
        "최종 운영 Threshold: 0.60",
        "Final operating threshold: 0.60",
    )
)


st.sidebar.caption(
    tr(
        "AI는 공정 recipe를 자동 변경하지 않습니다.",
        "The AI does not automatically change the process recipe.",
    )
)


# ============================================================
# 7. HEADER
# ============================================================

st.markdown(
    f"""
<div class="main-title">
AI–SPC Process Monitor
</div>

<div class="sub-title">
{tr(
    "현재 wafer까지 확보된 정보를 이용해 바로 다음 wafer의 SPC 이상 위험을 예측하는 사전경고 시스템",
    "An early-warning system that uses information available through the current wafer to predict SPC anomaly risk for the very next wafer"
)}
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# PAGE 1. LIVE MONITOR
# ============================================================

if page == "live":

    select1, select2, select3 = (
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


    with select1:

        selected_lot = st.selectbox(
            tr(
                "Lot 선택",
                "Select Lot"
            ),
            lot_list,
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


    with select2:

        selected_current_wafer = st.selectbox(
            tr(
                "현재 Wafer",
                "Current Wafer"
            ),
            wafer_list,
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


    pre_alarm_eligible = bool(
        int(
            selected_row[
                "pre_alarm_eligible"
            ]
        )
        == 1
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
{tr("현재", "Current")} W{int(selected_current_wafer)}
&nbsp;&nbsp;→&nbsp;&nbsp;
{tr("다음", "Next")} W{next_wafer} {tr("예측", "Prediction")}
</div>
""",
            unsafe_allow_html=True,
        )


    st.markdown(
        "---"
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


    risk = float(
        selected_row[
            "predicted_risk"
        ]
    )


    # --------------------------------------------------------
    # Card styles
    #
    # current_spc_status 실제 값:
    # NORMAL / ALARM
    # --------------------------------------------------------

    if current_spc_status == "ALARM":

        card1_class = (
            "alarm-card"
        )

        current_status_class = (
            "big-status-alarm"
        )

        current_status_display = tr(
            "🚨 이상",
            "🚨 ALARM"
        )


    else:

        card1_class = (
            "step-card"
        )

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


    card1, arrow1, card2, arrow2, card3 = (
        st.columns(
            [
                4,
                0.55,
                4,
                0.55,
                4
            ]
        )
    )


    # ========================================================
    # STEP 1
    # ========================================================

    with card1:

        if current_spc_status == "ALARM":

            current_description = tr(
                "현재 wafer가 SPC 관리한계를 벗어났습니다. 공정 상태 확인이 필요합니다.",
                "The current wafer is outside the SPC control limits. The process condition should be checked.",
            )


        else:

            current_description = tr(
                "현재 wafer의 실제 측정 결과를 기존 SPC 기준으로 확인합니다.",
                "The current wafer's measured result is checked against the conventional SPC limits.",
            )


        st.markdown(
            f"""
<div class="{card1_class}">

<div class="step-number">
STEP 1
</div>

<div class="step-title">
{tr("현재 공정 상태", "Current Process Status")}
</div>

<div class="{current_status_class}">
{current_status_display}
</div>

<div class="small-description">

{tr("현재", "Current")}
W{int(selected_current_wafer)}
{tr("평균 식각 깊이", "mean etch depth")}:

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

            prediction_text = tr(
                f"다음 W{next_wafer}에서 SPC 이상 가능성이 높다고 판단했습니다.",
                f"The AI estimates a high probability of an SPC anomaly on the next wafer, W{next_wafer}.",
            )


        elif system_status == "SPC ALARM":

            prediction_text = tr(
                "현재 SPC가 이미 이상 상태를 감지했습니다.",
                "Conventional SPC has already detected an abnormal condition.",
            )


        else:

            prediction_text = tr(
                f"다음 W{next_wafer}의 SPC 이상 위험이 낮다고 판단했습니다.",
                f"The AI estimates a low SPC anomaly risk for the next wafer, W{next_wafer}.",
            )


        st.markdown(
            f"""
<div class="{card2_class}">

<div class="step-number">
STEP 2
</div>

<div class="step-title">
{tr("AI 다음 Wafer 예측", "AI Next-Wafer Prediction")}
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

            action_title = tr(
                (
                    '<span style="white-space:nowrap;">'
                    '⚠ 엔지니어'
                    '</span><br>'
                    '<span style="white-space:nowrap;">'
                    '확인 필요'
                    '</span>'
                ),
                (
                    '<span style="white-space:nowrap;">'
                    '⚠ Engineer'
                    '</span><br>'
                    '<span style="white-space:nowrap;">'
                    'Review Required'
                    '</span>'
                ),
            )


            action_description = tr(
                "AI가 다음 wafer의 위험 신호를 감지했습니다. 주요 공정 센서를 확인한 뒤 공정 조정 여부를 엔지니어가 결정합니다.",
                "The AI detected a risk signal for the next wafer. The engineer reviews the key process sensors and decides whether process adjustment is needed.",
            )


        elif system_status == "SPC ALARM":

            action_title = tr(
                (
                    '<span style="white-space:nowrap;">'
                    '🚨 즉시 공정'
                    '</span><br>'
                    '<span style="white-space:nowrap;">'
                    '확인 필요'
                    '</span>'
                ),
                (
                    '<span style="white-space:nowrap;">'
                    '🚨 Immediate Process'
                    '</span><br>'
                    '<span style="white-space:nowrap;">'
                    'Review Required'
                    '</span>'
                ),
            )


            action_description = tr(
                "기존 SPC가 이미 이상을 감지했습니다. 엔지니어의 공정 점검이 필요합니다.",
                "Conventional SPC has already detected an anomaly. An engineer should inspect the process.",
            )


        else:

            action_title = tr(
                "✅ 계속 모니터링",
                "✅ Continue Monitoring"
            )


            action_description = tr(
                "현재 AI 사전경고는 없습니다. 다음 wafer까지 공정을 계속 모니터링합니다.",
                "There is no AI early warning at this point. Continue monitoring through the next wafer.",
            )


        st.markdown(
            f"""
<div class="{card3_class}">

<div class="step-number">
STEP 3
</div>

<div class="step-title">
{tr("최종 판단", "Final Decision")}
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
        and
        system_status == "EARLY WARNING"
    ):

        st.markdown(
            f"""
<div class="warning-banner">

⚠ <b>AI EARLY WARNING</b>

&nbsp;&nbsp;|&nbsp;&nbsp;

{tr(
    f"현재 W{int(selected_current_wafer)}는 SPC 기준 정상",
    f"Current W{int(selected_current_wafer)} is normal by SPC"
)}

&nbsp;&nbsp;→&nbsp;&nbsp;

{tr(
    f"다음 W{next_wafer} 이상 위험",
    f"Next W{next_wafer} anomaly risk"
)}

<span style="
font-size:1.35rem;
font-weight:950;
color:#ea580c;
">
{risk:.0%}
</span>

&nbsp;&nbsp;→&nbsp;&nbsp;

<b>
{tr(
    "엔지니어 확인 필요",
    "Engineer review required"
)}
</b>

</div>
""",
            unsafe_allow_html=True,
        )


    elif current_spc_status == "ALARM":

        st.markdown(
            f"""
<div class="alarm-banner">

🚨 <b>SPC ALARM</b>

&nbsp;&nbsp;|&nbsp;&nbsp;

{tr(
    f"현재 W{int(selected_current_wafer)}가 SPC 관리한계를 벗어났습니다.",
    f"Current W{int(selected_current_wafer)} is outside the SPC control limits."
)}

&nbsp;&nbsp;→&nbsp;&nbsp;

<b>
{tr(
    "즉시 공정 상태 확인 필요",
    "Immediate process review required"
)}
</b>

</div>
""",
            unsafe_allow_html=True,
        )


    else:

        st.success(
            tr(
                (
                    f"현재 W{int(selected_current_wafer)}는 정상이며, "
                    f"AI 역시 다음 W{next_wafer}의 위험을 낮게 평가했습니다."
                ),
                (
                    f"Current W{int(selected_current_wafer)} is normal, "
                    f"and the AI also estimates low risk for next W{next_wafer}."
                ),
            )
        )


    # --------------------------------------------------------
    # 평가 대상 여부 안내
    # --------------------------------------------------------

    if not pre_alarm_eligible:

        st.markdown(
            f"""
<div class="post-spc-banner">

ℹ️ <b>
{tr(
    "성능평가 제외 구간",
    "Excluded from early-warning evaluation"
)}
</b>

<br><br>

{tr(
    "이 시점은 해당 Lot에서 최초 SPC 이상이 이미 발생한 이후입니다. AI 위험도 자체는 계속 표시하지만, Recall·Precision·ROC-AUC 등 최종 사전경고 성능 계산에는 포함하지 않습니다.",
    "This point occurs after the first SPC anomaly in the lot. AI risk is still displayed, but the sample is excluded from final early-warning metrics such as Recall, Precision, and ROC-AUC."
)}

</div>
""",
            unsafe_allow_html=True,
        )


    # ========================================================
    # DETAILS
    # ========================================================

    with st.expander(
        tr(
            "🔎 상세 분석 보기",
            "🔎 View Detailed Analysis"
        )
    ):

        st.markdown(
            tr(
                "### 기존 SPC 기준",
                "### Conventional SPC Limits"
            )
        )


        spc1, spc2, spc3, spc4 = (
            st.columns(
                4
            )
        )


        spc1.metric(
            tr(
                "현재 식각 깊이",
                "Current etch depth"
            ),
            f"{selected_row['current_mean_si_etch']:.3f}",
        )


        spc2.metric(
            "LCL",
            f"{selected_row['LCL']:.3f}"
        )


        spc3.metric(
            tr(
                "중심선",
                "Center Line"
            ),
            f"{selected_row['CL']:.3f}"
        )


        spc4.metric(
            "UCL",
            f"{selected_row['UCL']:.3f}"
        )


        st.markdown(
            tr(
                "### AI 판단에 기여한 주요 공정 신호",
                "### Key Process Signals Contributing to the AI Decision",
            )
        )


        contributor_cols = (
            st.columns(
                3
            )
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


            with contributor_cols[
                rank - 1
            ]:

                if pd.isna(
                    process_name
                ):

                    st.info(
                        tr(
                            "표시할 공정 변수가 없습니다.",
                            "No process variable is available to display.",
                        )
                    )


                else:

                    st.markdown(
                        f"""
<div class="contributor-card">

<div class="contributor-rank">
#{rank}
{tr(
    "주요 공정 신호",
    "Key process signal"
)}
</div>

<div class="contributor-name">
{clean_process_name(process_name)}
</div>

{tr(
    "센서 요약값",
    "Sensor summary value"
)}:

<b>
{process_value:.4f}
</b>

<br>

{tr(
    "AI 기여도",
    "AI contribution"
)}:

<b>
{process_shap_value:+.3f}
</b>

</div>
""",
                        unsafe_allow_html=True,
                    )


        st.caption(
            tr(
                "SHAP은 AI가 어떤 공정 신호를 중요하게 사용했는지 설명합니다. 해당 변수가 실제 이상 원인이라는 뜻은 아닙니다.",
                "SHAP explains which process signals the AI relied on most. It does not prove that the variable is the physical cause of the anomaly.",
            )
        )


    # ========================================================
    # RETROSPECTIVE VALIDATION
    # ========================================================

    with st.expander(
        tr(
            "🧪 실제 다음 Wafer 결과 확인",
            "🧪 Check Actual Next-Wafer Result"
        )
    ):

        st.caption(
            tr(
                "다음 wafer 결과는 실제 예측 시점에는 알 수 없습니다. 연구에서 AI 예측이 맞았는지 확인하기 위한 사후 검증입니다.",
                "The next-wafer result is not available at prediction time. This section is retrospective validation used only to check whether the AI prediction was correct.",
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


        result_col1, result_col2 = (
            st.columns(
                2
            )
        )


        with result_col1:

            st.metric(
                tr(
                    f"실제 W{next_wafer} 식각 깊이",
                    f"Actual W{next_wafer} etch depth",
                ),
                f"{selected_row['next_actual_mean_si_etch']:.3f}",
            )


            st.write(
                tr(
                    "실제 SPC 상태:",
                    "Actual SPC status:"
                )
            )


            if actual_status == "SPC VIOLATION":

                st.error(
                    tr(
                        "🚨 SPC 이상 발생",
                        "🚨 SPC anomaly occurred"
                    )
                )


            else:

                st.success(
                    tr(
                        "🟢 정상",
                        "🟢 Normal"
                    )
                )


        with result_col2:

            st.write(
                tr(
                    "AI 예측 평가:",
                    "AI prediction evaluation:"
                )
            )


            result_message = (
                result_text(
                    warning_result
                )
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


            elif warning_result == "POST-FIRST-SPC":

                st.info(
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
                tr(
                    (
                        f"🎯 AI가 W{int(selected_current_wafer)} 시점에서 "
                        f"W{next_wafer} 위험을 경고했고, 실제로 "
                        f"W{next_wafer}에서 최초 SPC 이상이 발생했습니다."
                    ),
                    (
                        f"🎯 At W{int(selected_current_wafer)}, "
                        f"the AI warned about risk on W{next_wafer}, "
                        f"and the first SPC anomaly actually occurred "
                        f"on W{next_wafer}."
                    ),
                )
            )


# ============================================================
# PAGE 2. LOT TIMELINE
# ============================================================

elif page == "timeline":

    st.markdown(
        f"""
<div class="section-title">
{tr(
    "Lot 전체 흐름",
    "Lot-Level Timeline"
)}
</div>
""",
        unsafe_allow_html=True,
    )


    st.write(
        tr(
            "한 Lot 안에서 AI 위험도가 어떻게 변하고, 언제 실제 SPC 이상이 발생했는지 확인합니다.",
            "Track how AI risk changes within a lot and when the actual SPC anomaly occurs.",
        )
    )


    lot_list = sorted(
        system_df[
            "lot_number"
        ]
        .unique()
    )


    selected_lot = st.selectbox(
        tr(
            "Lot 선택",
            "Select Lot"
        ),
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


    fig = go.Figure()


    fig.add_trace(
        go.Scatter(

            x=
                lot_df[
                    "next_wafer"
                ],

            y=
                lot_df[
                    "predicted_risk"
                ],

            mode=
                "lines+markers",

            name=
                tr(
                    "AI 위험도",
                    "AI Risk"
                ),

            hovertemplate=(
                tr(
                    "예측 대상",
                    "Target"
                )
                +
                " W%{x}<br>"
                +
                tr(
                    "위험도",
                    "Risk"
                )
                +
                " %{y:.1%}<extra></extra>"
            ),
        )
    )


    # --------------------------------------------------------
    # FINAL THRESHOLD = 0.60
    # --------------------------------------------------------

    fig.add_hline(
        y=0.60,
        line_dash="dash",
        annotation_text=
            tr(
                "사전경고 기준 60%",
                "Early-warning threshold 60%",
            ),
    )


    warning_rows = (
        lot_df[
            (
                lot_df[
                    "ai_warning"
                ]
                == True
            )
            &
            (
                lot_df[
                    "pre_alarm_eligible"
                ]
                == 1
            )
        ]
        .copy()
    )


    if len(
        warning_rows
    ) > 0:

        fig.add_trace(
            go.Scatter(

                x=
                    warning_rows[
                        "next_wafer"
                    ],

                y=
                    warning_rows[
                        "predicted_risk"
                    ],

                mode=
                    "markers",

                marker=dict(
                    size=16,
                    symbol="triangle-up"
                ),

                name=
                    tr(
                        "AI 사전경고",
                        "AI Early Warning"
                    ),
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
            x=
                first_alarm_wafer,

            line_dash=
                "dot",

            annotation_text=
                tr(
                    f"최초 SPC 이상 W{first_alarm_wafer}",
                    f"First SPC anomaly W{first_alarm_wafer}",
                ),
        )


    fig.update_layout(

        title=
            tr(
                f"Lot {int(selected_lot)} — 다음 Wafer AI 위험도",
                f"Lot {int(selected_lot)} — AI Risk for the Next Wafer",
            ),

        xaxis_title=
            tr(
                "예측 대상 Wafer",
                "Target Wafer"
            ),

        yaxis_title=
            tr(
                "AI 위험도",
                "AI Risk"
            ),

        yaxis=dict(
            range=[
                0,
                1
            ],
            tickformat=".0%"
        ),

        height=480,

        hovermode=
            "x unified",
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    if len(
        first_alarm_rows
    ) > 0:

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
                tr(
                    (
                        f"🎯 Lot {int(selected_lot)}: "
                        f"W{current_wafer} 처리 후 AI가 "
                        f"W{target_wafer} 위험을 {risk_value:.0%}로 경고했고, "
                        f"실제로 W{target_wafer}에서 최초 SPC 이상이 발생했습니다."
                    ),
                    (
                        f"🎯 Lot {int(selected_lot)}: "
                        f"after W{current_wafer}, the AI warned of "
                        f"{risk_value:.0%} risk for W{target_wafer}, "
                        f"and the first SPC anomaly actually occurred "
                        f"on W{target_wafer}."
                    ),
                )
            )


        else:

            st.warning(
                tr(
                    (
                        f"Lot {int(selected_lot)}: "
                        f"W{target_wafer}에서 최초 SPC 이상이 발생했지만, "
                        f"직전 W{current_wafer}의 AI 위험도는 "
                        f"{risk_value:.0%}로 사전경고 기준에 미달했습니다."
                    ),
                    (
                        f"Lot {int(selected_lot)}: "
                        f"the first SPC anomaly occurred on W{target_wafer}, "
                        f"but the AI risk after W{current_wafer} was "
                        f"{risk_value:.0%}, below the early-warning threshold."
                    ),
                )
            )


    else:

        st.info(
            tr(
                f"Lot {int(selected_lot)}에서는 관측 가능한 wafer 구간 내 최초 SPC 이상이 확인되지 않았습니다.",
                f"No first SPC anomaly was observed within the available wafer range for Lot {int(selected_lot)}.",
            )
        )


    with st.expander(
        tr(
            "📋 상세 Lot 데이터 보기",
            "📋 View Detailed Lot Data"
        )
    ):

        display_df = (
            lot_df[
                [
                    "current_wafer",
                    "next_wafer",
                    "pre_alarm_eligible",
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


        display_df[
            "pre_alarm_eligible"
        ] = (
            display_df[
                "pre_alarm_eligible"
            ]
            .map(
                lambda x:
                    tr(
                        "평가 포함",
                        "Included"
                    )
                    if int(x) == 1
                    else
                    tr(
                        "평가 제외",
                        "Excluded"
                    )
            )
        )


        if LANG == "ko":

            rename_dict = {

                "current_wafer":
                    "현재 Wafer",

                "next_wafer":
                    "예측 대상",

                "pre_alarm_eligible":
                    "사전경고 평가",

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


        else:

            rename_dict = {

                "current_wafer":
                    "Current Wafer",

                "next_wafer":
                    "Prediction Target",

                "pre_alarm_eligible":
                    "Early-Warning Evaluation",

                "current_spc_status":
                    "Current SPC",

                "predicted_risk":
                    "AI Risk",

                "system_status":
                    "AI Decision",

                "next_actual_status":
                    "Actual Next Result",

                "warning_result":
                    "Evaluation",
            }


        display_df = (
            display_df
            .rename(
                columns=
                    rename_dict
            )
        )


        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# PAGE 3. MODEL VALIDATION
# ============================================================

elif page == "validation":

    st.markdown(
        f"""
<div class="section-title">
{tr(
    "모델 검증 결과",
    "Model Validation Results"
)}
</div>
""",
        unsafe_allow_html=True,
    )


    st.write(
        tr(
            "최종 XGBoost 모델의 조기경보 성능과 SPC baseline 및 모델 선택의 타당성을 확인합니다.",
            "Review the final XGBoost early-warning performance and the rationale for the SPC baseline and model selection.",
        )
    )


    # ========================================================
    # 1. FINAL PERFORMANCE
    # ========================================================

    st.markdown(
        tr(
            "### 1. 최종 AI Early-Warning 성능",
            "### 1. Final AI Early-Warning Performance",
        )
    )


    sample_count = int(
        get_metric(
            "pre_alarm_samples",
            0
        )
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


    f1 = get_metric(
        "f1"
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


    # --------------------------------------------------------
    # First row
    # --------------------------------------------------------

    metric_row1 = (
        st.columns(
            4
        )
    )


    metric_row1[
        0
    ].metric(
        tr(
            "평가 Sample",
            "Evaluation Samples"
        ),
        str(
            sample_count
        ),
    )


    metric_row1[
        1
    ].metric(
        tr(
            "이상 탐지율 (Recall)",
            "Recall"
        ),
        f"{recall:.1%}",
    )


    metric_row1[
        2
    ].metric(
        tr(
            "경고 정확도 (Precision)",
            "Precision"
        ),
        f"{precision:.1%}",
    )


    metric_row1[
        3
    ].metric(
        tr(
            "정상 판별률",
            "Specificity"
        ),
        f"{specificity:.1%}",
    )


    # --------------------------------------------------------
    # Second row
    # --------------------------------------------------------

    metric_row2 = (
        st.columns(
            4
        )
    )


    metric_row2[
        0
    ].metric(
        "F1 Score",
        f"{f1:.3f}",
    )


    metric_row2[
        1
    ].metric(
        "ROC-AUC",
        f"{roc_auc:.3f}",
    )


    metric_row2[
        2
    ].metric(
        tr(
            "오경고율",
            "False Warning Rate"
        ),
        f"{false_warning_rate:.1%}",
    )


    metric_row2[
        3
    ].metric(
        tr(
            "최초 이상 사전경고",
            "First-Anomaly Early Warning"
        ),
        f"{first_warning_rate:.1%}",
    )


    # --------------------------------------------------------
    # Small explanation instead of blue box
    # --------------------------------------------------------

    st.caption(
        tr(
            (
                "최종 성능은 LOLO 검증에서 최초 SPC 이상 발생 전 "
                "28개 sample을 대상으로 Threshold 0.60에서 평가했습니다. "
                "최초 이상 사전경고율 62.5%는 실제 최초 SPC 이상 8건 중 "
                "5건을 직전 wafer에서 경고한 event-level 지표입니다."
            ),
            (
                "Final performance was evaluated at Threshold 0.60 "
                "on 28 pre-first-SPC-anomaly samples using LOLO validation. "
                "The 62.5% first-anomaly early-warning rate is an event-level metric: "
                "5 of 8 first-SPC-anomaly events were warned one wafer ahead."
            ),
        )
    )


    st.markdown(
        "---"
    )


    # ========================================================
    # 2. BASELINE
    # ========================================================

    st.markdown(
        tr(
            "### 2. 왜 W1~W3를 SPC baseline으로 사용했나?",
            "### 2. Why use W1–W3 as the SPC baseline?",
        )
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


        baseline_show = (
            baseline_show
            .rename(
                columns={
                    "baseline_definition":
                        "Baseline",

                    "pooled_sigma":
                        tr(
                            "Pooled σ",
                            "Pooled σ"
                        ),

                    "three_sigma_half_width":
                        tr(
                            "±3σ 폭",
                            "±3σ half-width"
                        ),

                    "lots_with_spc_alarm":
                        tr(
                            "SPC alarm Lot",
                            "Lots with SPC alarm"
                        ),
                }
            )
        )


        st.dataframe(
            baseline_show,
            use_container_width=True,
            hide_index=True,
        )


        st.success(
            tr(
                "W1~W3에서 pooled σ와 lot 간 baseline 변동성이 가장 작았습니다. 반면 W4를 포함하면 drift가 진행되기 시작한 구간이 baseline에 포함되어 기준선이 영향을 받을 수 있었습니다.",
                "W1–W3 produced the smallest pooled σ and the lowest between-lot baseline variability. Including W4 can introduce the beginning of process drift into the baseline.",
            )
        )


    else:

        st.info(
            tr(
                "baseline_sensitivity_summary.csv를 results/tables 폴더에 추가하면 결과가 표시됩니다.",
                "Add baseline_sensitivity_summary.csv to results/tables to display the baseline validation results.",
            )
        )


    st.markdown(
        "---"
    )


    # ========================================================
    # 3. MODEL COMPARISON
    # ========================================================

    st.markdown(
        tr(
            "### 3. AI 모델 비교",
            "### 3. AI Model Comparison",
        )
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
                tr(
                    "오경고율",
                    "False warning"
                ),

            "warning_rate":
                tr(
                    "최초 이상 사전경고",
                    "First alarm warning"
                ),
        }


        model_show = (
            model_show
            .rename(
                columns=
                    rename_dict
            )
        )


        st.dataframe(
            model_show,
            use_container_width=True,
            hide_index=True,
        )


        st.info(
            tr(
                "여러 분류모델을 LOLO 방식으로 동일한 기본 threshold 0.50에서 비교한 뒤, 본 연구의 핵심 목표인 최초 SPC 이상 사전경고 성능을 고려하여 XGBoost를 최종 모델로 선정했습니다. 이후 XGBoost에 대해 threshold 분석을 수행하여 최종 운영 기준을 0.60으로 설정했습니다.",
                "Multiple classifiers were compared using LOLO validation at the same default threshold of 0.50. XGBoost was selected considering the study's main objective of warning before the first SPC anomaly, and a subsequent threshold analysis set the final operating threshold to 0.60.",
            )
        )


    else:

        st.info(
            tr(
                "model_comparison_summary.csv를 results/tables 폴더에 추가하면 모델 비교 결과가 표시됩니다.",
                "Add model_comparison_summary.csv to results/tables to display the model comparison.",
            )
        )


    st.markdown(
        "---"
    )


    # ========================================================
    # 4. THRESHOLD
    # ========================================================

    with st.expander(
        tr(
            "Threshold별 성능 비교 보기",
            "View Performance by Threshold",
        )
    ):

        threshold_fig = (
            go.Figure()
        )


        columns_to_plot = [

            (
                "recall",
                "Recall"
            ),

            (
                "false_warning_rate",
                tr(
                    "오경고율",
                    "False warning"
                ),
            ),

            (
                "first_alarm_warning_rate",
                tr(
                    "최초 이상 사전경고",
                    "First alarm warning"
                ),
            ),
        ]


        for (
            column_name,
            label
        ) in columns_to_plot:

            if column_name not in threshold_df.columns:
                continue


            threshold_fig.add_trace(
                go.Scatter(

                    x=
                        threshold_df[
                            "threshold"
                        ],

                    y=
                        threshold_df[
                            column_name
                        ],

                    mode=
                        "lines+markers",

                    name=
                        label,
                )
            )


        threshold_fig.add_vline(
            x=0.60,
            line_dash="dash",
            annotation_text=
                tr(
                    "최종 기준 0.60",
                    "Final threshold 0.60",
                ),
        )


        threshold_fig.update_layout(

            xaxis_title=
                tr(
                    "경고 임계값",
                    "Warning threshold",
                ),

            yaxis_title=
                tr(
                    "성능",
                    "Performance",
                ),

            yaxis=dict(
                range=[
                    0,
                    1
                ],
                tickformat=".0%"
            ),

            height=450,
        )


        st.plotly_chart(
            threshold_fig,
            use_container_width=True,
        )


        # ----------------------------------------------------
        # Threshold table
        # ----------------------------------------------------

        threshold_display_columns = [
            "threshold",
            "recall",
            "precision",
            "specificity",
            "f1",
            "false_warning_rate",
            "first_alarm_warned",
            "first_alarm_total",
        ]


        available_threshold_columns = [
            col
            for col in threshold_display_columns
            if col in threshold_df.columns
        ]


        threshold_show = (
            threshold_df[
                available_threshold_columns
            ]
            .copy()
        )


        for col in [
            "recall",
            "precision",
            "specificity",
            "false_warning_rate",
        ]:

            if col in threshold_show.columns:

                threshold_show[
                    col
                ] = (
                    threshold_show[
                        col
                    ]
                    .map(
                        lambda x:
                            f"{x:.1%}"
                    )
                )


        if "f1" in threshold_show.columns:

            threshold_show[
                "f1"
            ] = (
                threshold_show[
                    "f1"
                ]
                .map(
                    lambda x:
                        f"{x:.3f}"
                )
            )


        if (
            "first_alarm_warned"
            in threshold_show.columns
            and
            "first_alarm_total"
            in threshold_show.columns
        ):

            threshold_show[
                "First alarm"
            ] = (
                threshold_show[
                    "first_alarm_warned"
                ]
                .astype(int)
                .astype(str)
                +
                "/"
                +
                threshold_show[
                    "first_alarm_total"
                ]
                .astype(int)
                .astype(str)
            )


            threshold_show = (
                threshold_show
                .drop(
                    columns=[
                        "first_alarm_warned",
                        "first_alarm_total"
                    ]
                )
            )


        threshold_rename = {

            "threshold":
                "Threshold",

            "recall":
                "Recall",

            "precision":
                "Precision",

            "specificity":
                "Specificity",

            "f1":
                "F1",

            "false_warning_rate":
                tr(
                    "오경고율",
                    "False warning"
                ),
        }


        threshold_show = (
            threshold_show
            .rename(
                columns=
                    threshold_rename
            )
        )


        st.dataframe(
            threshold_show,
            use_container_width=True,
            hide_index=True,
        )


        st.caption(
            tr(
                (
                    "Threshold 0.60은 0.50과 동일한 Recall 62.5%와 "
                    "최초 SPC 이상 사전경고율 5/8을 유지하면서 "
                    "오경고율을 30.0%에서 25.0%로 낮춰 "
                    "최종 운영 기준으로 선정했습니다."
                ),
                (
                    "Threshold 0.60 was selected as the final operating threshold "
                    "because it maintained the same 62.5% recall and 5/8 "
                    "first-SPC-anomaly warning performance as 0.50 while reducing "
                    "the false-warning rate from 30.0% to 25.0%."
                ),
            )
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    "---"
)


st.markdown(
    f"""
<div class="disclaimer">

<b>
{tr(
    "연구용 Prototype",
    "Research Prototype"
)}
</b>

<br><br>

{tr(
    "본 시스템은 historical BOSCH plasma-etching 데이터를 wafer 순서대로 재생한 연구용 prototype입니다.",
    "This system is a research prototype that replays historical BOSCH plasma-etching data in wafer order."
)}

<br><br>

{tr(
    "최종 AI 모델은 현재 wafer까지 확보 가능한 이력·conditioning 정보와 training fold 내부에서 선정한 Top10 공정 feature를 이용해 바로 다음 wafer 한 장의 SPC 이상 위험을 예측합니다.",
    "The final AI model uses history and conditioning information available through the current wafer together with Top10 process features selected inside each training fold to predict SPC anomaly risk for exactly one next wafer."
)}

<br><br>

{tr(
    "최종 성능 평가는 최초 SPC 이상이 발생하기 전의 28개 sample(pre_alarm_eligible = 1)에 대해 LOLO 방식으로 수행했으며, 최종 운영 threshold는 0.60입니다.",
    "Final performance was evaluated using LOLO validation on 28 samples before the first SPC anomaly (pre_alarm_eligible = 1), with a final operating threshold of 0.60."
)}

<br><br>

{tr(
    "AI가 recipe를 자동 변경하지 않으며, 최종 공정 확인 및 조정 여부는 엔지니어가 판단합니다.",
    "The AI does not automatically change the recipe; the engineer makes the final decision on process review and adjustment."
)}

</div>
""",
    unsafe_allow_html=True,
)