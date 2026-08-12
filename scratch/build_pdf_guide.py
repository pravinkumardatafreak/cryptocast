import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY

def generate_pdf():
    pdf_filename = r"c:\Users\pravi\.antigravity\cryptocast\CryptoCast_Project_Guide.pdf"
    
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    # Custom styles
    primary_color = colors.HexColor("#0f172a") # Dark navy slate
    secondary_color = colors.HexColor("#1e3a8a") # Deep blue
    accent_color = colors.HexColor("#0284c7") # Sky blue accent
    text_dark = colors.HexColor("#334155")
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=primary_color,
        alignment=TA_CENTER,
        spaceAfter=8
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=accent_color,
        alignment=TA_CENTER,
        spaceAfter=15
    )
    
    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        textColor=secondary_color,
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=primary_color,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=text_dark,
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=text_dark,
        leftIndent=15,
        spaceAfter=4
    )

    code_style = ParagraphStyle(
        'Code_Custom',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#0f172a"),
        backColor=colors.HexColor("#f1f5f9"),
        borderColor=colors.HexColor("#cbd5e1"),
        borderWidth=0.5,
        borderPadding=5,
        spaceAfter=6
    )

    story = []

    # Title Banner
    story.append(Paragraph("CryptoCast: GUVI Live Evaluation Guide", title_style))
    story.append(Paragraph("GUVI × HCL Master Data Science Program — Complete Capstone Defense Blueprint", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=accent_color, spaceBefore=0, spaceAfter=15))

    # Section 1: GUVI Live Evaluation Schedule & Marking Matrix
    story.append(Paragraph("1. GUVI Live Evaluation & Scoring Matrix (Total: 60 Marks)", h1_style))
    
    # Schedule Table
    sched_data = [
        [Paragraph("<b>Process Step</b>", body_style), Paragraph("<b>Duration</b>", body_style), Paragraph("<b>Key Strategy / Focus Area</b>", body_style)],
        [Paragraph("1. Self Introduction", body_style), Paragraph("3 Mins", body_style), Paragraph("State name, background, domain interest (Quant Finance / Deep Learning).", body_style)],
        [Paragraph("2. Project Problem Statement & Explanation", body_style), Paragraph("12 Mins", body_style), Paragraph("Follow the mandatory GUVI 13-step model building sequence.", body_style)],
        [Paragraph("3. Mock Viva Questions", body_style), Paragraph("10 Mins", body_style), Paragraph("1-2 deep technical questions per technology (Python, PyTorch, SciPy, Streamlit).", body_style)],
        [Paragraph("4. Feedback & Evaluation Scoring", body_style), Paragraph("5 Mins", body_style), Paragraph("Review scoring notes and ZEN portal entry confirmation.", body_style)]
    ]
    
    t_sched = Table(sched_data, colWidths=[150, 60, 310])
    t_sched.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#e2e8f0")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_sched)
    story.append(Spacer(1, 10))

    # Marks Table
    marks_data = [
        [Paragraph("<b>Sl.</b>", body_style), Paragraph("<b>Evaluation Metric</b>", body_style), Paragraph("<b>Marks</b>", body_style), Paragraph("<b>How CryptoCast Achieves Top Marks</b>", body_style)],
        [Paragraph("1", body_style), Paragraph("Code Quality / Data Transformations", body_style), Paragraph("10", body_style), Paragraph("Leak-free scaling, stationary log returns, PEP-8 modular functions.", body_style)],
        [Paragraph("2", body_style), Paragraph("Proper Documentation (Git ReadMe/PPT)", body_style), Paragraph("10", body_style), Paragraph("100% committed ReadMe with 11 dashboard pages & multi-horizon tables.", body_style)],
        [Paragraph("3", body_style), Paragraph("Code Reusability (Modular Programming)", body_style), Paragraph("10", body_style), Paragraph("Decoupled `src/` directory with reusable strategy & model classes.", body_style)],
        [Paragraph("4", body_style), Paragraph("Presentation", body_style), Paragraph("10", body_style), Paragraph("Interactive 11-page Streamlit web app with custom glassmorphism dark theme.", body_style)],
        [Paragraph("5", body_style), Paragraph("Task Accomplishment", body_style), Paragraph("10", body_style), Paragraph("All 5 DL models trained & evaluated across 1D, 3D, and 7D horizons.", body_style)],
        [Paragraph("6", body_style), Paragraph("5 Mock Questions (Viva)", body_style), Paragraph("10", body_style), Paragraph("Clear answers for loss functions, RevIN, R² paradox, and stationarity.", body_style)],
        [Paragraph("", body_style), Paragraph("<b>TOTAL SCORE</b>", body_style), Paragraph("<b>60 Marks</b>", body_style), Paragraph("<b>Goal: 60 / 60</b>", body_style)]
    ]
    
    t_marks = Table(marks_data, colWidths=[25, 180, 55, 260])
    t_marks.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1e3a8a")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BACKGROUND', (0,-1), (-1,-1), colors.HexColor("#f8fafc")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t_marks)
    story.append(Spacer(1, 15))

    # Section 2: The Mandatory 13-Step Model Building Speech (12-Min Defense)
    story.append(Paragraph("2. Mandatory GUVI 13-Step Model Building Speech (12-Minute Script)", h1_style))
    story.append(Paragraph("Follow this exact step-by-step sequence during your 12-minute project explanation:", body_style))

    steps_13 = [
        ("1. Domain Intro (3 Lines)", "Cryptocurrency markets represent a highly volatile, 24/7 decentralized asset class. Price discovery is driven by micro-structural momentum, macro liquidity cycles, and halving supply shocks. Applying quantitative data science to crypto requires overcoming non-stationarity and extreme return tail-risk."),
        ("2. Problem Statement (1 Line)", "To design, train, and compare multi-output deep learning models that predict Bitcoin closing prices across 1-day, 3-day, and 7-day horizons simultaneously."),
        ("3. Data Cleaning & Preprocessing", "Cleaned ~5,000 daily records (2010–2024). Converted raw string prices with commas to float, parsed volume suffixes (K, M, B), and enforced chronological 80/20 train-test splitting. Crucially, MinMaxScaler was fit ONLY on training data to strictly prevent data leakage."),
        ("4. EDA Findings (Univariate/Bivariate/Multivariate)", "Prices are heavily right-skewed and non-stationary. Daily returns show high leptokurtosis (heavy tails). Seasonality analysis revealed a strong Turn-of-the-Month (TOM) liquidity effect with Q4 (days 23-31) producing highest average daily returns (+0.775%)."),
        ("5. Feature Engineering", "Engineered 3 deterministic whitepaper protocol features derived from Satoshi Nakamoto's code: `Block_Reward` (50->25->12.5->6.25->3.125), `Days_Since_Halving`, and `Halving_Progress` (0.0 to 1.0). Unlike reactive technical indicators (RSI/SMA), these are known ahead of time."),
        ("6. Statistical Significance & Tests", "Ran parametric (Student's t-test, Welch's t-test, 1-Way ANOVA) and non-parametric (Mann-Whitney U) hypothesis tests across halving epochs. Confirmed statistically significant return differences across halving regimes (p < 0.05)."),
        ("7. Target & Class Formulation", "Transformed raw prices into stationary log returns: r = ln(P[t+h] / P[t]). Created long/short directional targets for trading signal evaluation and directional accuracy score calculation."),
        ("8. Base Model Selection", "Chose 1D-CNN (3 Conv1D blocks with causal padding) as the initial benchmark model due to its rapid training speed and local pattern extraction capability."),
        ("9. Deep Learning Models Implemented", "Implemented 5 PyTorch architectures: 1D-CNN, Simple RNN, LSTM (gated memory highway), Transformer (multi-head self-attention), and PatchTST (patching + Reversible Instance Normalization RevIN). Used Adam optimizer, ReduceLROnPlateau, and gradient clipping (max_norm=1.0)."),
        ("10. Model Evaluation Metrics & Custom Loss", "Evaluated MAE, RMSE, and MAPE. Designed a custom `DirectionalMSELoss` combining MSE with a directional penalty (alpha=0.15) to penalize wrong trade direction predictions more heavily than magnitude errors."),
        ("11. Final Model Selection & Rationale", "PatchTST is selected as the champion model. It beats all models at 3D (MAPE 3.85%) and 7D (MAPE 5.93%) horizons while matching LSTM at 1D (MAPE 2.06%). RevIN eliminates distribution shift, and patching reduces token attention complexity by 96% (O(60^2) -> O(5^2))."),
        ("12. Conclusion & Feature Importance", "SHAP feature importance analysis confirms that `Block_Reward` and `Price` drive long-term structural forecasts. Halving-aligned Walk-Forward Validation confirmed model stability across unseen market regimes."),
        ("13. Business Suggestion & Solution", "Deployed a 3-state (Long/Short/Cash) Trading Simulator with dynamic position sizing on Page 10. Outperformed passive Buy-and-Hold by avoiding bear market drawdowns, providing actionable ROI for crypto hedge funds.")
    ]

    for title, desc in steps_13:
        story.append(Paragraph(f"<b>{title}</b>", h2_style))
        story.append(Paragraph(desc, body_style))

    story.append(Spacer(1, 10))

    # Section 3: Essential GUVI 15 EDA Questions & Answers
    story.append(Paragraph("3. GUVI 15 EDA Questions Cheat Sheet", h1_style))
    
    eda_q_a = [
        ("1. Basic Characteristics?", "4,964 daily observations across 9 feature columns (Aug 2010 to Mar 2024)."),
        ("2. Overall Structure?", "Time-series index with continuous floats for price/volume and discrete protocol values."),
        ("3. Patterns in Data?", "Exponential price growth interrupted by 4-year post-halving bull-bear cycles."),
        ("4. Presence of Outliers?", "Extreme daily price shocks (e.g. +25% or -30% single-day moves) present in return distribution."),
        ("5. Missing Values?", "Zero missing values in primary BTC dataset after cleaning raw Investing.com formatting."),
        ("6. Correctness of Data?", "Verified OHLC relationships (High >= Max(Open, Close), Low <= Min(Open, Close))."),
        ("7. Variable Correlations?", "Open, High, Low, and Close correlate near 0.99 with raw price; returns exhibit near zero autocorrelation."),
        ("8. Comparison to Past Performance?", "Post-2020 institutional era shows lower daily return variance compared to early 2011-2013 era."),
        ("9. Seasonality Present?", "Turn-of-the-Month (TOM) effect identified: Q4 (Days 23-31) and Q1 (Days 1-7) show higher returns."),
        ("10. Feature Variability?", "Daily volume varies across 6 orders of magnitude ($10K in 2010 to $50B+ in 2024)."),
        ("11. Discrepancies Observed?", "Raw prices fail stationarity tests (ADF test p > 0.05); log returns pass (ADF p < 0.001)."),
        ("12. Unexpected Results?", "Simple RNN outperforms Transformer at 7D horizon due to Transformer overfitting on small sequence samples."),
        ("13. Subset Behaviors?", "Pre-halving vs. post-halving subsets show statistically significant return variance differences."),
        ("14. Required Transformations?", "Log return transformation r = ln(P[t+h]/P[t]) and MinMaxScaler to range [0, 1]."),
        ("15. Gaps Identified?", "Lack of order book depth data; mitigated by incorporating macroeconomic liquidity proxies.")
    ]

    for q, a in eda_q_a:
        story.append(Paragraph(f"<b>Q: {q}</b> — {a}", bullet_style))

    story.append(Spacer(1, 10))

    # Section 4: Technology Mock Viva Q&A
    story.append(Paragraph("4. Technology Mock Viva Questions & Answers", h1_style))

    mock_qa = [
        ("Q1: Explain DirectionalMSELoss formula and why it's critical.", 
         "Loss = MSE + alpha * mean(ReLU(-y_pred * sign(y_true))). Standard MSE only measures magnitude error. In trading, predicting 'up' when market drops 'down' causes capital loss. The ReLU term adds extra penalty whenever signs disagree (alpha=0.15)."),
        
        ("Q2: What is the R² Paradox in financial time series?", 
         "Evaluating predicted vs actual raw prices gives R² ≈ 0.99 because yesterday's price is highly predictive of today's price level (random walk baseline). Evaluating predicted vs actual daily returns gives R² ≈ 1-3%. In quantitative finance, explaining 2% of daily return variance is considered highly successful."),
        
        ("Q3: How does PatchTST solve the attention bottleneck?", 
         "Standard Transformers treat each daily timestep as a token (60 tokens -> O(60^2) = 3600 attention operations). PatchTST groups 12 days into 1 patch (5 patches -> O(5^2) = 25 operations), reducing complexity by 96% while capturing local semantic sub-series."),
        
        ("Q4: Explain Reversible Instance Normalization (RevIN).", 
         "RevIN subtracts mean and divides by std of each input instance before feeding it to the network, and adds mean/std back after prediction. This handles non-stationary distribution shifts between training and live market regimes."),
        
        ("Q5: What is Walk-Forward Validation with Halving-Aligned Folds?", 
         "Standard k-fold cross-validation breaks time order (causing data leakage). WFV uses an expanding window that trains on past data and tests on future data. Halving-aligned folds ensure each test window covers an entire 4-year bull-bear halving epoch.")
    ]

    for q, a in mock_qa:
        story.append(Paragraph(f"<b>{q}</b>", h2_style))
        story.append(Paragraph(a, body_style))

    doc.build(story)
    print(f"Successfully generated {pdf_filename}")

if __name__ == '__main__':
    generate_pdf()
