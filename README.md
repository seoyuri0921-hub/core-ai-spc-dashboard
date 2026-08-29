# AI-SPC Early Warning for Semiconductor Etch Process Drift

> **SPC의 사후 이상 감지를 AI 기반 Next-Wafer 예측으로 보완한 반도체 식각 공정 조기경고 시스템**

<p align="center">
  <a href="https://core-ai-spc-dashboard-kyfg6yxrrfjntwabxcfqqf.streamlit.app/">
    <img src="https://img.shields.io/badge/▶%20LIVE%20DASHBOARD-OPEN-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white">
  </a>
</p>

<p align="center">
  <sub>※ Streamlit Community Cloud may require a short wake-up time after inactivity.</sub>
</p>

---

## Dashboard Preview

<p align="center">
  <img src="assets/dashboard_demo.gif"
       alt="AI-SPC Dashboard Demo"
       width="1000">
</p>

<p align="center">
  <b>Lot 3 · W6 → W7 Early Warning Demonstration</b>
</p>

---

## 1. Key Results

| Validation | Result | Engineering Meaning |
|---|---:|---|
| **Process Drift** | **-0.1188 µm / wafer** | Lot 초기 수준 차이를 통제한 후에도 유의한 하향 Drift 확인 |
| **Drift Significance** | **p < 0.001** | 반복 공정에 따른 Etch Depth 감소가 통계적으로 유의 |
| **Validation Strategy** | **Leave-One-Lot-Out** | 동일 Lot 정보가 Train/Test에 동시에 포함되는 Data Leakage 방지 |
| **First SPC Alarm Prediction** | **5 / 8 (62.5%)** | 최초 SPC 이상 8건 중 5건을 1-Wafer 앞서 경고 |
| **ROC-AUC** | **0.731** | Next-Wafer 이상 위험 분류 성능 |
| **Final Threshold** | **0.60** | Recall 유지하면서 False Alarm 감소 |
| **False Alarm Rate** | **25%** | Threshold 0.50 대비 30% → 25% 감소 |

### Core Result

> **현재 Wafer는 SPC 기준 정상이어도, AI가 다음 Wafer의 이상 위험을 먼저 경고할 수 있는가?**

대표 사례인 **Lot 3**에서는 현재 W6까지 SPC 기준 정상 상태였지만,  
AI는 다음 Wafer인 **W7의 SPC 이상 위험을 약 73%**로 예측했습니다.

이후 실제 W7에서 SPC 관리한계 이탈이 발생하여,

**AI Early Warning → 1 Wafer → SPC Alarm**

의 조기경고 사례를 확인했습니다.

---

## 2. Problem Definition

반도체 Plasma Etching 공정은 동일한 Recipe를 사용하더라도 Chamber 상태와 공정 이력에 따라 결과가 점진적으로 변할 수 있습니다.

본 데이터에서는 Wafer가 순차적으로 처리될수록 평균 Si Etch Depth가 감소하는 **Process Drift**가 확인되었습니다.

기존 SPC는 현재 Wafer의 측정값이 확보된 후에야 관리한계 이탈 여부를 판단할 수 있습니다.

따라서 본 프로젝트에서는 다음 질문을 정의했습니다.

> **현재까지 확보된 공정 정보만으로 다음 Wafer의 SPC 이상 위험을 미리 예측할 수 있는가?**

이를 위해 기존 SPC에 AI 기반 Next-Wafer 예측 기능을 결합했습니다.

---

## 3. Dataset

본 프로젝트는 Fraunhofer ENAS 및 Chemnitz University of Technology에서 공개한  
**BOSCH Plasma-Etching Dataset**을 사용했습니다.

### Process

- Bosch DRIE Process
- Etching Gas: `SF6`
- Passivation Gas: `C4F8`
- Wafer Diameter: `200 mm`
- Sequential Wafer Processing
- Conditioning 조건을 변경한 여러 Lot 구성

### Available Data

| Data | Description | Role |
|---|---|---|
| Wafer Measurement | 89-point Si / SiO2 Etch Depth | Drift 분석 및 SPC Target |
| Process Parameter | RF Power, Gas Flow, Pressure, Temperature 등 | AI Feature |
| Conditioning | Conditioning surface / repetition | Process History Feature |
| Wafer Order | W1, W2, W3 ... | Drift progression indicator |
| OES | 3648-channel Optical Emission Spectrum | Plasma State Analysis |

공정 센서 데이터는 5 Hz 시계열 데이터로 제공되며,  
Wafer 단위 분석을 위해 평균, 표준편차, 최솟값, 최댓값 등의 통계량으로 변환했습니다.

---

## 4. System Architecture

    Raw Etch / Process Data
              │
              ▼
    Data Preprocessing
              │
              ▼
    Wafer-level Feature Engineering
              │
              ├───────────────┐
              │               │
              ▼               ▼
       Drift Validation     SPC Baseline
                           W1-W3 / ±3σ
              │               │
              └───────┬───────┘
                      ▼
                Current Wafer Wn
                      │
                      ▼
                 XGBoost
                      │
                      ▼
           Next Wafer Wn+1 Risk
                      │
                      ▼
             Threshold = 0.60
                      │
              ┌───────┴────────┐
              ▼                ▼
           NORMAL        EARLY WARNING
                               │
                               ▼
                        SHAP Explanation
                               │
                               ▼
                        Engineer Review

### SPC와 AI의 역할

**SPC**

> 현재 Wafer가 정상 범위에 있는가?

**AI**

> 다음 Wafer가 SPC 관리한계를 벗어날 위험이 있는가?

따라서 AI가 SPC를 대체하는 것이 아니라,  
**SPC의 현재 상태 판단에 AI의 미래 위험 예측을 추가하는 구조**입니다.

---

## 5. Process Drift Validation

89-point Etch Depth 데이터로 각 Wafer의 평균 Si Etch Depth를 계산한 뒤,  
Wafer 처리 순서에 따른 변화를 분석했습니다.

Lot마다 초기 Etch Depth가 다르기 때문에 단순한 전체 회귀가 아니라  
Lot 차이를 통제한 ANCOVA를 수행했습니다.

### Result

    Etch Depth Change = -0.1188 µm / wafer
    p-value < 0.001
    R² = 0.792

즉, Lot별 초기 수준 차이를 고려한 이후에도 Wafer가 한 장씩 처리될수록  
평균 Si Etch Depth가 약 **0.119 µm 감소**하는 경향을 확인했습니다.

### Related Results

    results/tables/drift_ancova.csv
    results/tables/drift_lot_regression.csv

---

## 6. SPC Baseline Engineering

SPC 관리 기준을 임의로 설정하지 않고 초기 Wafer 범위에 따른 안정성을 비교했습니다.

### Baseline Sensitivity Analysis

| Baseline | Pooled σ | ±3σ Control Width | Lot-to-Lot Variation |
|---|---:|---:|---:|
| W1-W2 | 0.2141 | ±0.6423 µm | 0.1301 |
| **W1-W3** | **0.1775** | **±0.5324 µm** | **0.0812** |
| W1-W4 | 0.2284 | ±0.6851 µm | 0.0976 |

### Engineering Decision

**W1-W3를 최종 SPC Baseline으로 선정했습니다.**

- W1-W2는 Sample 수가 적어 개별 Wafer 변동에 민감
- W1-W3에서 Pooled σ가 가장 작음
- Lot 간 초기 변동성 역시 가장 안정적
- W1-W4에서는 이미 진행된 Drift 일부가 Baseline에 포함될 가능성 존재

따라서 각 Lot의 W1-W3 평균을 중심선으로 사용하고,  
**±3σ Control Limit**을 설정했습니다.

### Related Results

    results/tables/spc_baseline_comparison.csv
    results/tables/spc_baseline_summary.csv
    results/tables/baseline_sensitivity_summary.csv
    results/tables/baseline_sensitivity_center_shift.csv
    results/tables/shewhart_spc_detail.csv
    results/tables/shewhart_first_alarm.csv

---

## 7. Next-Wafer Prediction

AI 모델은 현재 Wafer `Wn`까지 확보된 정보만을 사용하여  
다음 Wafer `Wn+1`의 SPC 관리한계 이탈 위험을 예측하도록 설계했습니다.

### Input Features

- Current Wafer Etch Depth
- Wafer Uniformity
- Wafer Number
- Conditioning Information
- RF Power
- Gas Flow
- Pressure
- Temperature
- Backside Helium 관련 Signal
- 기타 Wafer-level Process Statistics

5 Hz Process Time-Series는 Wafer 단위로 다음 통계량을 생성했습니다.

    mean
    standard deviation
    minimum
    maximum

### Important Rule

다음 Wafer의 결과는 입력에 포함하지 않았습니다.

    Available at Prediction Time

    Current Wafer Information
    Previous Process History
    Current Process Sensor Summary

    Not Available at Prediction Time

    Next Wafer Etch Result
    Next Wafer Process Sensor Data

이를 통해 실제 공정에서 사용 가능한 정보 시점을 유지했습니다.

---

## 8. Leave-One-Lot-Out Validation

Wafer 단위 Random Split은 사용하지 않았습니다.

동일 Lot의 Wafer들은 서로 유사한 Chamber State와 Process History를 공유하므로,  
동일 Lot 데이터가 Train/Test에 함께 포함될 경우 성능이 실제보다 높게 평가될 수 있습니다.

따라서 **Leave-One-Lot-Out Cross Validation**을 적용했습니다.

    Fold 1
    Train : Lot 2 ~ Lot 10
    Test  : Lot 1

    Fold 2
    Train : Lot 1, Lot 3 ~ Lot 10
    Test  : Lot 2

    ...

    Repeat for every Lot

Feature Selection 역시 각 Training Fold 내부에서 독립적으로 수행하여  
Test Lot 정보가 Feature Selection에 사용되지 않도록 했습니다.

### Related Results

    results/tables/model_lolo_fold_metrics.csv
    results/tables/model_lolo_overall_metrics.csv
    results/tables/model_oof_predictions.csv
    results/tables/feature_selection_fold_rankings.csv
    results/tables/feature_selection_stability.csv

---

## 9. Model Comparison

동일한 Data와 LOLO Validation 조건에서 네 가지 분류 모델을 비교했습니다.

| Model | ROC-AUC | Recall | Precision | Specificity | False Alarm | First SPC Alarm |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.644 | 50.0% | 57.1% | 85.0% | 15% | 4/8 |
| Random Forest | 0.775 | 37.5% | 42.9% | 80.0% | 20% | 3/8 |
| SVM | **0.844** | 50.0% | **66.7%** | **90.0%** | **10%** | 4/8 |
| **XGBoost** | 0.731 | **62.5%** | 45.5% | 70.0% | 30% | **5/8** |

SVM은 전체 분류 성능에서는 가장 높은 ROC-AUC를 보였지만,  
본 프로젝트의 핵심 목적은 **SPC 이상이 발생하기 전에 위험을 최대한 사전 포착하는 것**이었습니다.

XGBoost는 실제 최초 SPC 이상 8건 중 5건을 직전 Wafer에서 사전경고해  
가장 높은 **First-Alarm Early Warning Recall**을 보였습니다.

따라서 최종 모델로 XGBoost를 선정했습니다.

### Related Results

    results/tables/model_comparison_summary.csv
    results/tables/model_comparison_first_alarm.csv
    results/tables/model_comparison_oof_predictions.csv
    results/tables/model_early_warning_metrics.csv

---

## 10. Threshold Engineering

AI Risk가 어느 수준일 때 실제 경고를 발생시킬 것인지 결정하기 위해  
Threshold Sensitivity Analysis를 수행했습니다.

| Threshold | Recall | False Alarm Rate | F1 Score | First SPC Alarm |
|---:|---:|---:|---:|---:|
| 0.30 | 87.5% | 35.0% | 0.636 | 7/8 |
| 0.40 | 62.5% | 35.0% | 0.500 | 5/8 |
| 0.50 | 62.5% | 30.0% | 0.526 | 5/8 |
| **0.60** | **62.5%** | **25.0%** | **0.556** | **5/8** |
| 0.70 | 37.5% | 15.0% | 0.429 | 3/8 |

Threshold를 높일수록 False Alarm은 감소하지만 Recall도 감소했습니다.

특히 `0.50 → 0.60`에서는

    First Alarm Recall : 5/8 → 5/8
    False Alarm Rate   : 30% → 25%

으로 조기경고 성능을 유지하면서 오경보를 감소시킬 수 있었습니다.

따라서 최종 운영 Threshold를 **0.60**으로 선정했습니다.

> Threshold는 모델 자체의 고정 성능이 아니라 공정 운영 목적에 따라 조정 가능한 Engineering Parameter로 해석했습니다.

### Related Results

    results/tables/final_threshold_analysis.csv
    results/tables/final_evaluation_metrics.csv
    results/tables/final_first_alarm_performance.csv

---

## 11. SHAP-Based Explainability

AI가 단순히 Risk Score만 제공하면 엔지니어가 실제 공정 점검에 활용하기 어렵습니다.

따라서 SHAP을 적용하여 각 예측에서 Risk 증가에 크게 기여한 Process Signal을 함께 제공합니다.

### Example

    Lot 3
    Current Wafer : W6
    Next Wafer    : W7

    AI Risk
    73%

    Top Process Signal #1
    Helium BP Flow (Mean)

    Top Process Signal #2
    Platen RF Reflected Power (Variation)

SHAP 분석에서 다음과 같은 공정 관련 변수들이 여러 LOLO Fold에서 반복적으로 선택되었습니다.

- Heater Temperature
- RF Reflected Power
- Gas Flow
- Backside Helium Flow
- Pressure-related Features

다만 SHAP Importance는 **예측에 사용된 중요도**를 의미하며,  
해당 변수가 Drift의 직접적인 Root Cause임을 의미하지는 않습니다.

따라서 SHAP은 자동 원인 판정이 아니라

> **AI 경고 발생 시 엔지니어가 우선 확인할 Process Signal**

로 활용했습니다.

### Related Results

    results/tables/shap_global_importance.csv
    results/tables/shap_process_importance.csv
    results/tables/shap_process_interpretation.csv
    results/tables/shap_process_risk_direction.csv
    results/tables/shap_first_alarm_case_contributors.csv
    results/tables/shap_successful_warning_process.csv

---

## 12. AI-SPC Dashboard

분석 결과를 실제 공정 모니터링 상황에서 확인할 수 있도록 Streamlit 기반 Dashboard를 구현했습니다.

### Dashboard Features

#### Real-Time Monitor

- Current Lot / Wafer 선택
- 현재 Wafer SPC 상태 확인
- Next-Wafer AI Risk 확인
- NORMAL / EARLY WARNING 상태 표시
- SHAP 기반 주요 Process Signal 확인

#### Lot Timeline

- Wafer 진행에 따른 Risk 변화
- AI Early Warning 시점
- 실제 SPC Alarm 시점 비교

#### Model Validation

- ROC-AUC
- Recall
- Precision
- False Alarm Rate
- Threshold Trade-off
- First SPC Alarm Early Warning Performance

### Run Dashboard

    pip install -r requirements.txt
    streamlit run 16_streamlit_dashboard.py

---

## 13. Repository Structure

    core-ai-spc-dashboard/
    │
    ├── data/
    │   └── processed/
    │       ├── next_wafer_dataset.csv
    │       ├── process_etch_matching.csv
    │       ├── process_feature_list.csv
    │       ├── process_features_clean.csv
    │       ├── process_features_full.csv
    │       ├── process_only_wafers.csv
    │       ├── wafer_process_selected.csv
    │       ├── wafer_process_table.csv
    │       └── wafer_table.csv
    │
    ├── results/
    │   └── tables/
    │       ├── drift_ancova.csv
    │       ├── drift_lot_regression.csv
    │       ├── spc_baseline_comparison.csv
    │       ├── shewhart_spc_detail.csv
    │       ├── model_comparison_summary.csv
    │       ├── model_lolo_overall_metrics.csv
    │       ├── final_threshold_analysis.csv
    │       ├── final_first_alarm_performance.csv
    │       ├── shap_global_importance.csv
    │       ├── shap_process_importance.csv
    │       └── ...
    │
    ├── 16_streamlit_dashboard.py
    ├── requirements.txt
    └── README.md

`results/tables/`에는 분석 과정에서 생성된 SPC, Drift, LOLO Validation, Model Comparison, Threshold 및 SHAP 결과를 저장했습니다.

---

## 14. Engineering Decisions

본 프로젝트에서는 단순한 모델 성능 향상보다 **공정 엔지니어링 관점의 의사결정**을 중요하게 고려했습니다.

| Problem | Analysis | Engineering Decision |
|---|---|---|
| SPC Baseline 범위 | W1-W2 / W1-W3 / W1-W4 비교 | **W1-W3 선정** |
| Lot 간 Data Leakage | Wafer Random Split 위험 확인 | **LOLO Validation 적용** |
| Model Selection | ROC-AUC와 Early Warning 성능 비교 | **XGBoost 선정** |
| Warning Threshold | Recall / False Alarm Trade-off 비교 | **0.60 선정** |
| AI Black Box | SHAP Feature Contribution 분석 | **Engineer Review 정보로 제공** |

---

## 15. What I Learned

이 프로젝트를 통해 머신러닝 모델의 높은 Accuracy 자체보다 제조 공정에서는 다음 요소가 더 중요하다는 점을 확인했습니다.

### 1. Correct Problem Definition

이미 발생한 이상을 분류하는 것이 아니라

> **현재 정보만으로 다음 Wafer의 위험을 예측**

하도록 예측 시점을 명확히 정의했습니다.

### 2. Validation Design Matters

Wafer 단위 Random Split보다 실제 공정 구조를 반영한 **Lot 단위 Validation**이 중요했습니다.

### 3. Model Metric ≠ Engineering Objective

ROC-AUC가 가장 높은 모델이 반드시 가장 적합한 모델은 아니었습니다.

본 시스템에서는 **First SPC Alarm을 얼마나 미리 포착하는가**가 더 중요한 목표였습니다.

### 4. Threshold Is an Engineering Trade-off

Threshold는 단순한 분류 기준이 아니라

**Missed Alarm ↔ False Alarm**

사이의 공정 운영 Trade-off였습니다.

### 5. AI Should Support Engineer Decisions

SHAP을 통해 AI의 판단 근거를 제공하되,  
공정 원인을 자동으로 단정하지 않고 최종 판단은 엔지니어에게 남기는 구조로 설계했습니다.

---

## 16. Limitations

본 프로젝트는 다음과 같은 한계를 가지고 있습니다.

- 공개 연구 데이터 기반 분석
- 약 10개 Lot 규모로 데이터 규모가 제한적
- 실제 Fab 생산라인에서의 Real-Time Validation 미수행
- First SPC Alarm 사례가 8건으로 제한적
- SHAP 결과는 예측 기여도이며 공정 인과관계를 직접 증명하지 않음

따라서 본 결과를 실제 양산 수율 개선 효과로 직접 해석하지 않았습니다.

향후 실제 공정 데이터가 축적된다면

- Lot / Chamber 데이터 확대
- Chamber 상태 변수 추가
- Real-Time Sensor Streaming
- Online SPC + AI Monitoring
- Threshold 운영 최적화

등으로 확장할 수 있습니다.

---

## 17. Future Work

    Process Tool
         │
         ▼
    Real-Time Sensor Streaming
         │
         ▼
    Wafer Feature Aggregation
         │
         ├──────────────┐
         ▼              ▼
    Real-Time SPC    AI Prediction
         │              │
         └──────┬───────┘
                ▼
           Early Warning
                │
                ▼
           SHAP Evidence
                │
                ▼
          Engineer Review

궁극적으로는 SPC가 현재 공정 상태를 판단하고 AI가 미래 위험을 예측하는  
**SPC-AI Collaborative Process Monitoring System**으로 확장하는 것이 목표입니다.

---

## 18. Tech Stack

### Data Analysis

![Python](https://img.shields.io/badge/Python-3.x-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-blue)
![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-blue)

### Machine Learning

![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Classification-red)
![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-green)

### Visualization / Deployment

![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![GitHub](https://img.shields.io/badge/GitHub-Portfolio-black)

---

## 19. Project Summary

> **SPC는 현재 Wafer의 이상을 판단하고, AI는 다음 Wafer의 위험을 예측한다.**

본 프로젝트는 반도체 식각 공정에서 발생하는 Drift를 데이터로 검증하고,  
SPC 기반 이상 기준과 AI 기반 Next-Wafer Prediction을 결합하여  
기존 사후 감지 중심 공정관리를 **선제적 Early Warning 방식으로 확장할 가능성**을 제시했습니다.

### Final Result

    10 / 10 Lots
    Etch Depth Decreasing Trend

            ↓

    SPC Baseline
    W1-W3 / ±3σ

            ↓

    LOLO Validation

            ↓

    XGBoost Next-Wafer Prediction

            ↓

    5 / 8 First SPC Alarms
    Predicted 1 Wafer Earlier

            ↓

    Threshold 0.60
    False Alarm Rate 25%

            ↓

    SHAP-based
    Engineer Decision Support

---

## Author

Semiconductor Process Engineering & Data Analytics Portfolio

**Focus**

`Semiconductor Process Engineering` · `SPC` · `Process Data Analytics` · `Machine Learning` · `AI-based Manufacturing Monitoring`
