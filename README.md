# Data-Driven Shipping Mode Optimization using Prescriptive Analytics

An end-to-end data science and operations research pipeline that minimizes expected logistics delay costs on real-world transaction data. The project leverages a hybrid architecture: **Predictive Modeling** (4 mode-specific Logistic Regression classifiers) to estimate delay probabilities, combined with **Prescriptive Analytics** (Mixed-Integer Linear Programming via PuLP) to find global cost-optimal routing solutions.

---

## 🚀 Business Impact & Key Results

* **45.79% Expected Cost Savings:** Shifting from legacy heuristics to mathematical optimization slashed expected delay penalties by nearly half.
* **Proactive Risk Mitigation:** By evaluating multi-class shipment outcomes across all options simultaneously, the model successfully rerouted **over 23% of at-risk orders** to safer modes without breaching systemic capacity constraints.
* **Strategic Mode Flipping:** Transitioned over-allocated, expensive options (*First Class*, *Second Class*) toward high-probability *Standard Class* pipelines, while surgically upgrading critically bottlenecked routes to *Same Day* execution.

---

## 🛠️ Tech Stack & Library Frameworks
* **Predictive ML:** `scikit-learn` (ColumnTransformers, OneHotEncoder, Balanced Multinomial Logistic Regression)
* **Optimization (MILP):** `PuLP` (Linear Programming API with default COIN-OR CBC Solver)
* **Exploratory Data Analysis:** `pandas`, `numpy`
* **Data Visualization:** `seaborn`, `matplotlib`, `plotly` (Dynamic Sankey Diagrams)

---

## 📈 Methodology & Architecture

### 1. Advanced Preprocessing & Engineering
* Cleaned real-world delivery state tables, filtering out administrative noise (`ON_HOLD` orders).
* Extracted localized high-dimensional temporal variables (`order_month`, `order_dayofweek`, `order_weekofyear`) and captured seasonal spikes via market features (`peak_season`).
* Synthesized composite interaction variables (`month_mode`, `region_mode`) to catch localized carrier patterns.

### 2. Segmented Predictive Modeling
* Engineered a global data preprocessing architecture via `ColumnTransformer` to handle continuous scaling and categorical sparse encodings.
* Rather than a static classification strategy, the system trains **4 distinct mode-specific multinomial models** with cost-sensitive learning (`class_weight='balanced'`) to counteract delivery target imbalance.
* Dynamically extracts cross-class risk vectors to compile an extensive, row-by-row delay probability matrix across all potential alternatives.

### 3. Prescriptive Mixed-Integer Optimization
The problem is formalized as a classical assignment routing problem to minimize total expected risk penalties subject to deterministic capacity limits.

**Mathematical Formulation:**

* **Decision Variable:** Let $x_{o,m} \in \{0, 1\}$ be $1$ if order $o$ is assigned to shipping mode $m$, and $0$ otherwise.
* **Objective Function:** Minimize total penalization where $P_{o,m}$ is the estimated failure probability and $W_m$ is the penalty constraint weight vector ($Standard: 1.0, Second: 1.2, First: 1.5, Same\ Day: 2.0$):
$$\min \sum_{o \in \text{Orders}} \sum_{m \in \text{Modes}} x_{o,m} \cdot (P_{o,m} \cdot W_m)$$

* **Constraints:** Ensure each individual order is exclusively routed through exactly one shipping configuration:
$$\sum_{m \in \text{Modes}} x_{o,m} = 1 \quad \forall o \in \text{Orders}$$

---
