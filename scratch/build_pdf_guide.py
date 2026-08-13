"""
CryptoCast — Comprehensive GUVI Live Evaluation & Viva Defense Guide
=====================================================================
Generates CryptoCast_Project_Guide.pdf with:
  1. GUVI 60-Mark Scoring Matrix & Evaluation Schedule
  2. Mandatory 13-Step Model Building Speech Script
  3. Code Walkthrough (module-by-module explanation)
  4. Fundamental Deep Learning Interview Q&A (Epoch, Backprop, Positional Encoding, etc.)
  5. Project-Specific Viva Q&A
  6. GUVI 15 EDA Questions Cheat Sheet
"""
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
    HRFlowable, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

def generate_pdf():
    pdf_filename = r"c:\Users\pravi\.antigravity\cryptocast\CryptoCast_Project_Guide.pdf"

    doc = SimpleDocTemplate(
        pdf_filename, pagesize=letter,
        rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # -- Colour Palette --
    C_NAVY   = colors.HexColor("#0f172a")
    C_BLUE   = colors.HexColor("#1e3a8a")
    C_SKY    = colors.HexColor("#0284c7")
    C_SLATE  = colors.HexColor("#334155")
    C_BG     = colors.HexColor("#f8fafc")
    C_BORDER = colors.HexColor("#cbd5e1")
    C_LGRAY  = colors.HexColor("#e2e8f0")

    # -- Styles --
    title_s = ParagraphStyle('T', parent=styles['Heading1'], fontName='Helvetica-Bold',
                             fontSize=22, leading=26, textColor=C_NAVY, alignment=TA_CENTER, spaceAfter=6)
    sub_s   = ParagraphStyle('S', parent=styles['Normal'], fontName='Helvetica-Bold',
                             fontSize=11, leading=14, textColor=C_SKY, alignment=TA_CENTER, spaceAfter=12)
    h1_s    = ParagraphStyle('H1', parent=styles['Heading1'], fontName='Helvetica-Bold',
                             fontSize=15, leading=18, textColor=C_BLUE, spaceBefore=14, spaceAfter=6, keepWithNext=True)
    h2_s    = ParagraphStyle('H2', parent=styles['Heading2'], fontName='Helvetica-Bold',
                             fontSize=11, leading=14, textColor=C_NAVY, spaceBefore=8, spaceAfter=3, keepWithNext=True)
    body_s  = ParagraphStyle('B', parent=styles['BodyText'], fontName='Helvetica',
                             fontSize=9, leading=12.5, textColor=C_SLATE, spaceAfter=4, alignment=TA_JUSTIFY)
    code_s  = ParagraphStyle('C', parent=styles['Normal'], fontName='Courier',
                             fontSize=8, leading=10.5, textColor=C_NAVY, backColor=colors.HexColor("#f1f5f9"),
                             borderColor=C_BORDER, borderWidth=0.5, borderPadding=4, spaceAfter=4)
    bullet_s = ParagraphStyle('BL', parent=styles['Normal'], fontName='Helvetica',
                              fontSize=8.5, leading=12, textColor=C_SLATE, leftIndent=12, spaceAfter=3)
    qa_q_s   = ParagraphStyle('QQ', parent=styles['Normal'], fontName='Helvetica-Bold',
                              fontSize=9, leading=12.5, textColor=C_BLUE, spaceBefore=6, spaceAfter=1)
    qa_a_s   = ParagraphStyle('QA', parent=styles['Normal'], fontName='Helvetica',
                              fontSize=8.5, leading=12, textColor=C_SLATE, leftIndent=10, spaceAfter=4)

    # Helpers
    def tbl(data, widths, header_bg=C_BLUE):
        t = Table(data, colWidths=widths)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), header_bg),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('GRID', (0,0), (-1,-1), 0.5, C_BORDER),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ]))
        return t

    def hdr(text): return [Paragraph(f"<b>{text}</b>", body_s)]
    def cell(text): return [Paragraph(text, body_s)]
    def qa(q, a): return [Paragraph(f"Q: {q}", qa_q_s), Paragraph(f"A: {a}", qa_a_s)]

    S = []  # story

    # ================================================================
    # TITLE
    # ================================================================
    S.append(Paragraph("CryptoCast: GUVI Live Evaluation &amp; Viva Defense Guide", title_s))
    S.append(Paragraph("GUVI × HCL Master Data Science Program — Complete Capstone Defense Blueprint", sub_s))
    S.append(HRFlowable(width="100%", thickness=1.5, color=C_SKY, spaceAfter=12))

    # ================================================================
    # SECTION 1: GUVI EVALUATION SCHEDULE & 60-MARK MATRIX
    # ================================================================
    S.append(Paragraph("1. GUVI Live Evaluation Schedule &amp; 60-Mark Scoring Matrix", h1_s))
    S.append(Paragraph("Your evaluation is split into four sequential phases. Hit each one with confident, structured delivery.", body_s))

    sched = [
        [Paragraph("<b>Phase</b>", body_s), Paragraph("<b>Duration</b>", body_s), Paragraph("<b>Key Strategy</b>", body_s)],
        [Paragraph("Self Introduction", body_s), Paragraph("3 min", body_s), Paragraph("Name, background, why Quant Finance + Deep Learning.", body_s)],
        [Paragraph("Project Explanation (13 Steps)", body_s), Paragraph("12 min", body_s), Paragraph("Follow the GUVI 13-step model building sequence below.", body_s)],
        [Paragraph("Mock Viva Questions", body_s), Paragraph("10 min", body_s), Paragraph("1-2 deep technical Qs per technology area.", body_s)],
        [Paragraph("Feedback &amp; Scoring", body_s), Paragraph("5 min", body_s), Paragraph("Evaluator enters marks on ZEN portal.", body_s)],
    ]
    S.append(tbl(sched, [150, 55, 320], C_LGRAY))
    S.append(Spacer(1, 8))

    marks = [
        [Paragraph("<b>Sl.</b>", body_s), Paragraph("<b>Criterion</b>", body_s), Paragraph("<b>Marks</b>", body_s), Paragraph("<b>How CryptoCast Scores 10/10</b>", body_s)],
        [Paragraph("1", body_s), Paragraph("Code Quality / Data Transformations", body_s), Paragraph("10", body_s), Paragraph("Stationary log returns, leak-free scaling, PEP-8, modular functions.", body_s)],
        [Paragraph("2", body_s), Paragraph("Documentation (Git ReadMe / PPT)", body_s), Paragraph("10", body_s), Paragraph("Comprehensive README with 11 pages, 5 models, ablation study, protocol features.", body_s)],
        [Paragraph("3", body_s), Paragraph("Code Reusability (Modular Design)", body_s), Paragraph("10", body_s), Paragraph("Decoupled src/ with 11 reusable modules; each page imports from shared utils.", body_s)],
        [Paragraph("4", body_s), Paragraph("Presentation / UI", body_s), Paragraph("10", body_s), Paragraph("11-page Streamlit dark-theme dashboard, glassmorphism cards, Plotly interactives.", body_s)],
        [Paragraph("5", body_s), Paragraph("Task Accomplishment", body_s), Paragraph("10", body_s), Paragraph("All 5 DL models trained across 3 horizons; WFV, SHAP, Live Prediction, Trading Sim.", body_s)],
        [Paragraph("6", body_s), Paragraph("5 Mock Questions (Viva)", body_s), Paragraph("10", body_s), Paragraph("Prepared answers for loss functions, RevIN, R² paradox, stationarity, WFV.", body_s)],
        [Paragraph("", body_s), Paragraph("<b>TOTAL</b>", body_s), Paragraph("<b>60</b>", body_s), Paragraph("<b>Target: 60 / 60</b>", body_s)],
    ]
    S.append(tbl(marks, [25, 170, 40, 290]))
    S.append(Spacer(1, 6))

    # ================================================================
    # SECTION 2: 13-STEP MODEL BUILDING SPEECH
    # ================================================================
    S.append(PageBreak())
    S.append(Paragraph("2. Mandatory GUVI 13-Step Model Building Speech (12-Minute Script)", h1_s))
    S.append(Paragraph("Follow this exact sequence during your 12-minute project walkthrough:", body_s))

    steps = [
        ("Step 1 — Domain Introduction (3 lines max)",
         "Cryptocurrency markets are a highly volatile, 24/7, decentralised asset class. Price discovery is driven by micro-structural momentum, "
         "macro liquidity cycles, and Bitcoin's deterministic halving supply shocks. Applying deep learning requires overcoming non-stationarity and extreme tail-risk."),
        ("Step 2 — Problem Statement (1 sentence)",
         "Design, train, and compare five multi-output deep learning architectures that forecast Bitcoin closing prices across 1-day, 3-day, and 7-day horizons simultaneously."),
        ("Step 3 — Data Cleaning &amp; Preprocessing",
         "Cleaned ~5,000 daily records (Aug 2010 – Mar 2024). Parsed raw string prices with commas to float, volume suffixes (K/M/B) to integers. "
         "Enforced chronological 80/20 train-test split. MinMaxScaler fit ONLY on training data to prevent data leakage."),
        ("Step 4 — EDA Findings (Univariate / Bivariate / Multivariate)",
         "Prices are heavily right-skewed and non-stationary. Daily returns show high leptokurtosis (fat tails). "
         "Turn-of-the-Month effect identified: Q4 (days 23–31) avg daily return +0.775%, Q1 (days 1–7) highest win rate 51.45%."),
        ("Step 5 — Feature Engineering",
         "Engineered 3 deterministic whitepaper protocol features from Satoshi Nakamoto's design: Block_Reward (50→25→12.5→6.25→3.125), "
         "Days_Since_Halving, and Halving_Progress (0.0 → 1.0). Unlike reactive indicators (RSI/SMA), these are known ahead of time."),
        ("Step 6 — Statistical Significance &amp; Tests",
         "Ran parametric (Student's t-test, Welch's t-test, 1-Way ANOVA) and non-parametric (Mann-Whitney U) hypothesis tests across halving epochs. "
         "Confirmed statistically significant return differences across regimes (p &lt; 0.05)."),
        ("Step 7 — Target &amp; Class Formulation",
         "Transformed raw prices into stationary log returns: r = ln(P[t+h] / P[t]). Created long/short directional targets for trading signal evaluation."),
        ("Step 8 — Base Model Selection",
         "Selected 1D-CNN (3 Conv1D blocks with causal padding) as the initial benchmark due to fast training and local pattern extraction."),
        ("Step 9 — Deep Learning Models Implemented",
         "Implemented 5 PyTorch architectures: 1D-CNN, Simple RNN, LSTM (gated memory), Transformer (multi-head self-attention), "
         "and PatchTST (patching + RevIN). Used Adam optimiser, ReduceLROnPlateau scheduler, gradient clipping (max_norm=1.0)."),
        ("Step 10 — Evaluation Metrics &amp; Custom Loss",
         "Evaluated MAE, RMSE, MAPE. Designed DirectionalMSELoss = MSE + alpha * mean(ReLU(-y_pred * sign(y_true))) with alpha=0.30 "
         "to penalise wrong trade direction predictions more heavily than magnitude errors."),
        ("Step 11 — Final Model Selection &amp; Rationale",
         "PatchTST is the champion model. Beats all at 3D (MAPE 3.74%) and 7D (MAPE 5.93%), matches LSTM at 1D (MAPE 2.06%). "
         "RevIN eliminates distribution shift; patching reduces attention complexity by 96% (O(60²) → O(5²))."),
        ("Step 12 — Conclusion &amp; Feature Importance",
         "SHAP feature importance confirms Block_Reward and Price drive long-term structural forecasts. "
         "Halving-aligned Walk-Forward Validation confirmed model stability across unseen market regimes."),
        ("Step 13 — Business Suggestion &amp; Solution",
         "Deployed a Dual-Directional (BUY &amp; SELL) Trading Simulator on Page 10. Trades execute when top 3 AI models (LSTM, Transformer, PatchTST) "
         "agree on 1D/3D/7D direction, but market price moves counter-trend on Monday/Tuesday (buying discounts below weekly open, selling premiums above weekly open). "
         "Outperformed passive Buy-and-Hold by avoiding drawdown crashes and generating two-way alpha."),
    ]
    for title, desc in steps:
        S.append(Paragraph(f"<b>{title}</b>", h2_s))
        S.append(Paragraph(desc, body_s))

    # ================================================================
    # SECTION 3: CODE WALKTHROUGH
    # ================================================================
    S.append(PageBreak())
    S.append(Paragraph("3. Code Walkthrough — Module-by-Module Explanation", h1_s))
    S.append(Paragraph("Use this section during your live code walkthrough. Open each file and explain these key lines.", body_s))

    code_modules = [
        ("src/step1_eda.py — Data Loading, Cleaning &amp; EDA",
         [
            "• Lines 45–70: Loads raw CSV from Investing.com, parses string prices (commas removed), volume suffixes (K→1000, M→1e6, B→1e9).",
            "• Lines 100–140: Computes whitepaper protocol features: Block_Reward = 50 / 2^epoch, Days_Since_Halving, Halving_Progress.",
            "• Lines 200–220: MinMaxScaler.fit() called ONLY on training partition → scaler.transform() on test. This is critical leak prevention.",
            "• Lines 250–300: Generates 8 EDA visualisation PNGs (price trend, volume, OHLC, distributions, correlation heatmap, rolling stats).",
            "• Output: scaled_data.npy, scalers.pkl, meta.json, data/btc_data.csv, visualizations/*.png.",
         ]),
        ("src/models.py — Centralized PyTorch Model Architectures &amp; Custom Loss",
         [
            "• CNN1D: 3 Conv1d layers with causal padding and AdaptiveAvgPool1d for local pattern extraction.",
            "• RNNModel &amp; LSTMModel: Stacked recurrent models for sequence temporal dependencies.",
            "• TransformerModel: Multi-head self-attention encoder (4 heads, d_model=256).",
            "• RevIN: Reversible Instance Normalization to eliminate non-stationary magnitude shifts.",
            "• PatchTSTModel: State-of-the-art patch tokenization + RevIN + Transformer encoder.",
            "• DirectionalMSELoss: MSE + alpha * mean(ReLU(-y_pred * sign(y_true))), alpha=0.30.",
         ]),
        ("src/train_model_pytorch.py — Model Training Loop &amp; Pipeline",
         [
            "• Lines 33–37: Reproducibility seeds (np.random.seed(42), torch.manual_seed(42), torch.cuda.manual_seed_all(42)).",
            "• Lines 69–99: create_sequences_multi() — Sliding window generator producing (X, y) where y contains log returns for 1D/3D/7D.",
            "• Lines 121–123: DataLoader with shuffle=False — Critical for time-series to prevent future data leaking into training batches.",
            "• Imports architectures cleanly from src.models (PEP-8 / DRY modular design).",
            "• Lines 390–432: Training loop — model.train(), zero_grad, forward, loss.backward(), clip_grad_norm_(max_norm=1.0), optimizer.step().",
            "• Lines 454–458: Price reconstruction — P_predicted = P_anchor * exp(r_predicted). This is the inverse log return transform.",
         ]),
        ("src/step4_wfv.py — Walk-Forward Validation Engine",
         [
            "• Imports architectures cleanly from src.models without code duplication.",
            "• Uses an expanding window: Train on epochs 0–1, test on epoch 2; then train on epochs 0–2, test on epoch 3.",
            "• Each fold retrains the model from scratch on the expanding window to avoid data contamination.",
            "• Results saved to wfv_results.json with per-fold, per-horizon, per-model metrics.",
         ]),
        ("src/stacked_meta_features.py — XGBoost / LightGBM Meta-Learner",
         [
            "• Stage 1: Extracts out-of-fold predictions from 3 deep learning base models (LSTM, Transformer, PatchTST).",
            "• Stage 2: Trains XGBoost or LightGBM on base model predictions as features to predict Weekly Directional Bias (Bullish/Bearish).",
            "• This is a classic Stacking Ensemble (Wolpert, 1992) approach.",
         ]),
        ("src/hypothesis_strategy.py — Statistical Hypothesis Engines",
         [
            "• run_weekly_hypothesis_test(): Splits BTC returns into Monday/Tuesday 'strategy' days vs other days, runs t-test and Mann-Whitney U.",
            "• run_daily_hypothesis_test(): Uses Z-score oversold threshold to identify entry signals, tests alpha significance.",
         ]),
        ("pages/7_Live_Prediction.py — Real-Time Inference",
         [
            "• Fetches last 60 days of BTC data from Yahoo Finance API (yfinance).",
            "• Computes all 9 features (OHLCV + Change% + Block_Reward + Days_Since_Halving + Halving_Progress).",
            "• Applies saved MinMaxScaler (scalers.pkl) transform, feeds tensor through PyTorch model.eval() + torch.no_grad().",
            "• Reconstructs target price: P_target = P_current * exp(r_predicted).",
         ]),
        ("pages/10_Trading_Simulator.py — Dual-Directional AI Strategy Simulator",
         [
            "• Evaluates Top 3 AI Models (LSTM, Transformer, PatchTST) across 1D/3D/7D horizons.",
            "• BUY ENTRY (LONG): Top 3 models predict higher prices across 1D/3D/7D, BUT price trades below weekly open on Mon/Tue.",
            "• SELL ENTRY (SHORT): Top 3 models predict lower prices across 1D/3D/7D, BUT price trades above weekly open on Mon/Tue.",
            "• Computes Sharpe Ratio, Max Drawdown, Win Rate, and compares vs Buy-and-Hold baseline.",
         ]),
    ]
    for mod_title, points in code_modules:
        S.append(Paragraph(f"<b>{mod_title}</b>", h2_s))
        for pt in points:
            S.append(Paragraph(pt, bullet_s))

    # ================================================================
    # SECTION 4: FUNDAMENTAL DL INTERVIEW Q&A
    # ================================================================
    S.append(PageBreak())
    S.append(Paragraph("4. Fundamental Deep Learning &amp; ML Interview Questions", h1_s))
    S.append(Paragraph("These are the basic concepts evaluators frequently test. Memorise the one-liner and the deeper explanation.", body_s))

    fundamentals = [
        ("What is an Epoch?",
         "One epoch = one complete forward and backward pass through the ENTIRE training dataset. In CryptoCast we use 10 epochs "
         "(optimised for fast live demo; convergence is achieved because pre-trained weights are saved). More epochs risk overfitting on small datasets."),

        ("What is a Batch and Batch Size?",
         "A batch is a subset of training samples processed together before one weight update. Batch size = 64 in CryptoCast. "
         "Larger batches → smoother gradients but more memory. Smaller batches → noisier gradients but better generalisation (implicit regularisation)."),

        ("What is Backpropagation?",
         "Backpropagation is the algorithm that computes gradients of the loss function with respect to each weight in the network using the chain rule of calculus. "
         "It propagates the error signal backward from the output layer to the input layer. In PyTorch, loss.backward() triggers this computation automatically via autograd."),

        ("What is Gradient Descent?",
         "Gradient descent is the optimisation algorithm that updates model weights in the direction that reduces the loss: w_new = w_old - lr * gradient. "
         "We use Adam optimiser (Adaptive Moment Estimation) which maintains per-parameter learning rates using first and second moment estimates of gradients."),

        ("What is the Vanishing Gradient Problem?",
         "When gradients are multiplied through many layers during backpropagation, they can become extremely small (vanish to ~0), "
         "causing early layers to stop learning. LSTMs solve this with gating mechanisms (forget gate, input gate, output gate) that create a 'memory highway' "
         "allowing gradients to flow unimpeded across long sequences."),

        ("What is the Exploding Gradient Problem?",
         "The opposite of vanishing: gradients grow exponentially large, causing weight updates to diverge. "
         "We solve this with gradient clipping: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) caps the gradient norm."),

        ("What is a Learning Rate?",
         "The step size for weight updates. Too high → overshooting the minimum; too low → stuck or slow convergence. "
         "CryptoCast uses lr=0.001 for CNN/RNN and lr=0.0005 for LSTM/Transformer/PatchTST. "
         "ReduceLROnPlateau halves the LR if validation loss plateaus for 4 epochs."),

        ("What is Positional Encoding?",
         "Transformers process all tokens in parallel (no inherent sequence order). Positional encoding injects position information. "
         "Standard Transformers use sinusoidal functions: PE(pos,2i) = sin(pos/10000^(2i/d_model)). "
         "PatchTST uses learnable position embeddings (nn.Parameter) instead, which are trained end-to-end with the model."),

        ("What is Self-Attention?",
         "Self-attention computes a weighted sum of all positions in a sequence, where the weights (attention scores) are derived from "
         "Query-Key dot products: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V. Multi-head attention runs this computation "
         "in parallel across 4 heads (in CryptoCast), each capturing different temporal patterns."),

        ("What is Dropout?",
         "Dropout randomly zeroes a fraction of neuron activations during training (p=0.2 in CryptoCast), forcing the network to learn redundant "
         "representations. During inference (model.eval()), all neurons are active. This is a regularisation technique to prevent overfitting."),

        ("What is Overfitting vs Underfitting?",
         "Overfitting: model memorises training data noise → low train loss, high validation loss. "
         "Underfitting: model is too simple to capture patterns → high loss on both. "
         "CryptoCast prevents overfitting via dropout, early stopping (patience=3), and gradient clipping."),

        ("What is Early Stopping?",
         "Stop training when validation loss stops improving for 'patience' consecutive epochs (patience=3 in CryptoCast). "
         "The best model weights (lowest val_loss) are saved and restored. This prevents overfitting on the training set."),

        ("What is a Loss Function?",
         "A mathematical function that quantifies the error between predicted and actual values. "
         "Standard: MSE = mean((y_pred - y_true)²). CryptoCast's custom DirectionalMSELoss adds a sign penalty to MSE, "
         "making the model care about predicting the correct trade direction, not just magnitude."),

        ("What is an Activation Function?",
         "A non-linear function applied after each layer to introduce non-linearity. Without it, stacking layers is equivalent to a single linear layer. "
         "ReLU(x) = max(0, x) is used in CryptoCast's feed-forward layers. PatchTST uses GELU (smoother approximation of ReLU). "
         "Sigmoid and Tanh are used inside LSTM gates."),

        ("What is the difference between RNN, LSTM, and Transformer?",
         "RNN: Processes sequence step-by-step; suffers from vanishing gradients on long sequences. "
         "LSTM: Adds gated memory cells (forget/input/output gates) to selectively remember or forget information. "
         "Transformer: Replaces recurrence with self-attention, processing all positions in parallel. Captures long-range dependencies without vanishing gradients."),

        ("What is MinMaxScaler and why fit only on training data?",
         "MinMaxScaler scales features to [0, 1] range: x_scaled = (x - x_min) / (x_max - x_min). "
         "If we fit on the full dataset, x_min and x_max contain information from the test set → data leakage. "
         "Fitting ONLY on training data ensures the model never sees test-set statistics during training."),

        ("What is Stationarity and why does it matter?",
         "A stationary time series has constant mean, variance, and autocorrelation over time. "
         "Raw BTC prices are non-stationary (strong upward trend). ML models assume i.i.d. (independent, identically distributed) data. "
         "Log returns r = ln(P[t+1]/P[t]) are approximately stationary, satisfying this assumption."),

        ("What is MAPE and why is it useful for financial forecasting?",
         "MAPE = Mean Absolute Percentage Error = mean(|actual - predicted| / |actual|) × 100%. "
         "Unlike MAE (in dollars), MAPE is scale-invariant. A 2% MAPE means the same whether BTC is at $1,000 or $60,000. "
         "This makes it the primary comparison metric across different price regimes."),

        ("What is R² Score and the R² Paradox?",
         "R² measures the proportion of variance explained by the model. Price-level R² ≈ 0.99 because yesterday's price "
         "predicts ~99% of today's price level (random walk baseline). Return-level R² ≈ 1–3%. In quantitative finance, "
         "explaining 2% of daily return variance is considered highly successful due to the noise-dominated, fractal nature of markets."),

        ("What is Walk-Forward Validation (WFV)?",
         "Standard k-fold cross-validation shuffles data → breaks time order → causes data leakage. WFV uses an expanding window: "
         "train on past, test on future. CryptoCast's Halving-Aligned WFV ensures each test fold covers a full 4-year halving epoch, "
         "testing model robustness across fundamentally different market regimes."),

        ("What is SHAP (SHapley Additive exPlanations)?",
         "SHAP assigns each feature a contribution value (Shapley value from game theory) for a specific prediction. "
         "Positive SHAP value → feature pushes prediction higher; negative → pushes lower. "
         "CryptoCast's SHAP analysis shows Block_Reward and Price as top drivers for long-horizon forecasts."),

        ("What is RevIN (Reversible Instance Normalisation)?",
         "RevIN normalises each input instance (subtract mean, divide by std) before the encoder, then reverses the operation after prediction. "
         "This handles distribution shift between training-era and live-era data (e.g., bull market training, bear market inference). "
         "It is a key reason PatchTST generalises better than standard Transformers."),

        ("What is Patching in PatchTST?",
         "Instead of treating each daily timestep as a separate token (60 tokens → O(60²) = 3,600 attention operations), "
         "PatchTST groups 12 consecutive days into 1 patch (5 patches → O(5²) = 25 operations). "
         "This is a 96% reduction in attention complexity while preserving local semantic sub-series information."),
    ]

    for q, a in fundamentals:
        S.extend(qa(q, a))

    # ================================================================
    # SECTION 5: PROJECT-SPECIFIC VIVA Q&A
    # ================================================================
    S.append(PageBreak())
    S.append(Paragraph("5. Project-Specific Viva Questions &amp; Answers", h1_s))

    project_qa = [
        ("Why predict log returns instead of raw prices?",
         "Raw prices are non-stationary (violate ML assumptions). Log returns are stationary, scale-invariant, and symmetric "
         "(+5% and -5% have equal magnitude in log space). This is the standard in quantitative finance."),

        ("Why PyTorch over TensorFlow/Keras?",
         "PyTorch uses dynamic computation graphs (eager execution), making custom architectures like RevIN and DirectionalMSELoss "
         "cleaner to debug. It is the dominant framework in modern research papers and top ML conferences."),

        ("Explain the DirectionalMSELoss formula and alpha=0.30.",
         "Loss = MSE + 0.30 * mean(ReLU(-y_pred * sign(y_true))). The ReLU fires only when prediction and actual have opposite signs. "
         "alpha=0.30 doubles the directional gradient penalty vs the original 0.15, improving 3D MAPE from 3.85% to 3.74%."),

        ("Why a 60-day sequence lookback window?",
         "60 days (~2 months) captures short-term momentum, weekly seasonality, and monthly cycles without introducing "
         "quadratic attention overhead. It also aligns with common technical analysis windows (50-day / 60-day moving averages)."),

        ("What are the 3 data leakage prevention measures?",
         "1. Chronological splitting with shuffle=False. 2. MinMaxScaler fit ONLY on training partition. "
         "3. Halving-Aligned Walk-Forward Validation using expanding windows across distinct 4-year epochs."),

        ("Why is 1D-CNN the weakest architecture?",
         "1D-CNN uses fixed kernel size 3 that captures only local 3-day patterns. It lacks global temporal memory "
         "for longer horizons compared to LSTMs (gated memory) or Transformers (self-attention across all positions)."),

        ("How does the Live Prediction engine work (Page 7)?",
         "Fetches last 60 days via Yahoo Finance API → computes all 9 features → applies saved MinMaxScaler.transform() "
         "→ feeds tensor through model.eval() + torch.no_grad() → reconstructs price via P = P_current * exp(r_predicted)."),

        ("What is the Turn-of-the-Month (TOM) effect you discovered?",
         "Q4 (Days 23–31) shows highest avg daily return +0.775% due to month-end capital reallocation and paycheck reinvestment. "
         "Q1 (Days 1–7) shows highest daily win rate 51.45%. Q2 (Days 8–15) is the weakest period at +0.195%."),

        ("How many Bitcoins are mined per halving cycle?",
         "Each epoch mines exactly 210,000 blocks × block_reward. Epoch 0: 10,500,000 BTC (50%), Epoch 1: 5,250,000 (25%), "
         "Epoch 2: 2,625,000 (12.5%), Epoch 3: 1,312,500 (6.25%), Epoch 4 (current): 656,250 (3.125%)."),

        ("What statistical tests did you run and what were the results?",
         "Student's t-test, Welch's t-test (unequal variance), Mann-Whitney U (non-parametric), and 1-Way ANOVA. "
         "All confirmed statistically significant return differences across halving regimes (p &lt; 0.05), "
         "validating that our protocol features capture real structural market shifts."),
    ]
    for q, a in project_qa:
        S.extend(qa(q, a))

    # ================================================================
    # SECTION 6: GUVI 15 EDA QUESTIONS
    # ================================================================
    S.append(PageBreak())
    S.append(Paragraph("6. GUVI 15 EDA Questions Cheat Sheet", h1_s))

    eda_qa = [
        ("1. Basic Characteristics?", "4,964 daily observations, 9 feature columns (Aug 2010 – Mar 2024)."),
        ("2. Overall Structure?", "Time-series index with continuous floats for OHLCV and discrete protocol values (Block_Reward)."),
        ("3. Patterns in Data?", "Exponential price growth interrupted by 4-year post-halving bull-bear cycles."),
        ("4. Presence of Outliers?", "Extreme daily price shocks (+25% / -30% single-day moves) present in return distribution tails."),
        ("5. Missing Values?", "Zero missing values after cleaning. Raw Investing.com formatting issues resolved in step1_eda.py."),
        ("6. Correctness of Data?", "Verified OHLC relationships: High ≥ max(Open, Close), Low ≤ min(Open, Close)."),
        ("7. Variable Correlations?", "OHLC prices correlate ~0.99 with each other. Returns exhibit near-zero autocorrelation (efficient market)."),
        ("8. Comparison to Past?", "Post-2020 institutional era shows lower daily return variance than early 2011–2013 retail era."),
        ("9. Seasonality Present?", "Turn-of-the-Month effect: Q4 (Days 23–31) and Q1 (Days 1–7) show highest returns."),
        ("10. Feature Variability?", "Volume varies across 6 orders of magnitude ($10K in 2010 to $50B+ in 2024)."),
        ("11. Discrepancies?", "Raw prices fail ADF stationarity test (p &gt; 0.05); log returns pass (p &lt; 0.001)."),
        ("12. Unexpected Results?", "Simple RNN outperforms Transformer at 7D horizon due to Transformer overfitting on small sample."),
        ("13. Subset Behaviours?", "Pre-halving vs post-halving subsets show statistically significant return variance differences."),
        ("14. Required Transformations?", "Log return transformation r = ln(P[t+h]/P[t]) and MinMaxScaler to [0, 1]."),
        ("15. Gaps Identified?", "Lack of order book depth data; mitigated by incorporating macro liquidity proxies and protocol features."),
    ]
    for q, a in eda_qa:
        S.append(Paragraph(f"<b>Q: {q}</b> — {a}", bullet_s))

    # ================================================================
    # SECTION 7: QUICK COMMANDS
    # ================================================================
    S.append(Spacer(1, 12))
    S.append(Paragraph("7. Quick Commands for Live Demo", h1_s))
    cmds = [
        [Paragraph("<b>Action</b>", body_s), Paragraph("<b>Command</b>", body_s)],
        [Paragraph("Launch Dashboard", body_s), Paragraph("<font face='Courier'>bun run dev</font>  (or <font face='Courier'>streamlit run app.py --server.port 8502</font>)", body_s)],
        [Paragraph("Run Full Pipeline", body_s), Paragraph("<font face='Courier'>bun run pipeline</font>  (or <font face='Courier'>python cryptocast.py</font>)", body_s)],
        [Paragraph("Run Compile Tests", body_s), Paragraph("<font face='Courier'>python scratch/test_all_pages.py</font>", body_s)],
        [Paragraph("Run WFV Only", body_s), Paragraph("<font face='Courier'>bun run wfv</font>", body_s)],
    ]
    S.append(tbl(cmds, [120, 405]))

    # Build
    doc.build(S)
    print(f"Successfully generated: {pdf_filename}")

if __name__ == '__main__':
    generate_pdf()
