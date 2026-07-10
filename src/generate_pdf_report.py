"""
CryptoCast - PDF Report Generator
=================================
Compiles the final GUVI Capstone Project Report into a professional PDF.
Reads the actual model metrics from model_comparison_results.csv and includes:
  1. Problem Statement & Business Use Cases
  2. Preprocessing & Data Leakage Prevention details
  3. Deep Learning Model Architectures (1D-CNN, RNN, LSTM, Transformer)
  4. Performance Comparison (dynamic table of MAE, RMSE, MAPE)
  5. Validation Strategy (Chronological Split + Walk-Forward demonstration)
  6. Business Interpretation & Conclusions

Usage:
    python src/generate_pdf_report.py
"""

import os
import sys
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(PROJECT_DIR, 'model_comparison_results.csv')
OUTPUT_PDF = os.path.join(PROJECT_DIR, 'CryptoCast_Project_Report.pdf')

def build_pdf():
    print("Generating CryptoCast Project Report PDF...")
    
    # Setup document
    doc = SimpleDocTemplate(
        OUTPUT_PDF,
        pagesize=letter,
        rightMargin=54, leftMargin=54,
        topMargin=54, bottomMargin=54
    )
    
    # Styles
    styles = getSampleStyleSheet()
    
    # Custom Styles for a premium look
    title_style = ParagraphStyle(
        'CoverTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor('#1a5276'),
        alignment=1, # Center
        spaceAfter=15
    )
    
    subtitle_style = ParagraphStyle(
        'CoverSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#566573'),
        alignment=1, # Center
        spaceAfter=30
    )
    
    h1_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        textColor=colors.HexColor('#1a5276'),
        spaceBefore=15,
        spaceAfter=10,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        'SubsectionHeader',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#2e86c1'),
        spaceBefore=10,
        spaceAfter=6,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=8
    )
    
    viva_tip_style = ParagraphStyle(
        'VivaTip',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor('#78281f'),
        backColor=colors.HexColor('#fdebd0'),
        borderPadding=6,
        spaceBefore=8,
        spaceAfter=12
    )
    
    story = []
    
    # --- Title Page Header ---
    story.append(Spacer(1, 20))
    story.append(Paragraph("CryptoCast: Multi-Horizon Bitcoin Price Forecasting", title_style))
    story.append(Paragraph("A Comparative Evaluation of Deep Learning Architectures (1D-CNN, RNN, LSTM, Transformer)", subtitle_style))
    story.append(Paragraph("<b>Domain:</b> Financial Analytics / Cryptocurrency / Deep Learning<br/><b>Institution:</b> GUVI Capstone Project evaluation", subtitle_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("<hr/>", body_style))
    story.append(Spacer(1, 10))
    
    # --- 1. Problem Understanding & Use Cases ---
    story.append(Paragraph("1. Problem Statement & Business Context", h1_style))
    p1 = ("Bitcoin prices are highly volatile and influenced by complex temporal patterns. "
          "Traditional statistical models (like ARIMA) struggle to capture non-linear relationships. "
          "This project builds and evaluates deep learning architectures to forecast Bitcoin's Close price "
          "over multiple horizons: 1-Day (1D), 3-Day (3D), and 7-Day (7D) ahead, using the past 60 days of historical data.")
    story.append(Paragraph(p1, body_style))
    
    story.append(Paragraph("Business Use Cases:", h2_style))
    use_cases = (
        "• <b>Crypto Trading Platforms:</b> Provide short-term price trend signals for active users.<br/>"
        "• <b>Algorithmic Trading Systems:</b> Inform automated order execution strategies across multiple horizons.<br/>"
        "• <b>Risk Management Tools:</b> Predict upcoming periods of extreme price volatility to adjust leverage/exposure.<br/>"
        "• <b>Investment Portfolios:</b> Supply investors with data-driven predictive insights for decision support."
    )
    story.append(Paragraph(use_cases, body_style))
    
    # --- 2. Data Preparation & Leakage Fix ---
    story.append(Paragraph("2. Data Preparation Strategy & Data Leakage Prevention", h1_style))
    p2 = ("The dataset contains daily records of Bitcoin from 2010 to 2024. As mandated by project guidelines, "
          "features are strictly restricted to <b>Open Price</b> and <b>Close Price (Price)</b>, representing a highly "
          "focused univariate/bivariate modeling framework.")
    story.append(Paragraph(p2, body_style))
    
    story.append(Paragraph("Crucial Viva Focus: Avoiding Data Leakage in Preprocessing", h2_style))
    leakage_desc = (
        "A common pitfall in time-series modeling is fitting data scalers (like MinMaxScaler) on the entire dataset "
        "before splitting it. This leaks future test set information (specifically min/max values) into training. "
        "In this updated version, the <b>MinMaxScaler is fitted solely on the training partition (first 80%)</b>, "
        "and that fitted scaler is used to transform both the train and test splits, guaranteeing mathematically "
        "valid out-of-sample metrics."
    )
    story.append(Paragraph(leakage_desc, body_style))
    story.append(Paragraph("<b>Viva Question Prep:</b> If asked how you prevented leakage, explain: 'We split our chronological data first, called <i>fit</i> on the training set only, and then transformed the full dataset using those pre-calculated parameters.'", viva_tip_style))
    
    # --- 3. Deep Learning Architectures ---
    story.append(Paragraph("3. Deep Learning Model Architectures", h1_style))
    
    story.append(Paragraph("1D Convolutional Neural Network (1D-CNN)", h2_style))
    story.append(Paragraph("Utilizes 3 stacked causal Conv1D layers (64, 64, and 32 filters, kernel size = 3) followed by Global Average Pooling, Dropout (0.2), and Dense layers. Causal convolutions prevent future sequence leakage, making them highly effective for extracting local, short-term trends quickly.", body_style))
    
    story.append(Paragraph("Simple Recurrent Neural Network (RNN)", h2_style))
    story.append(Paragraph("Consists of 2 stacked SimpleRNN layers (64 and 32 units) with recurrent dropout. Serves as a baseline recurrent network, though vulnerable to vanishing gradient issues over the 60-day historical window.", body_style))
    
    story.append(Paragraph("Long Short-Term Memory (LSTM)", h2_style))
    story.append(Paragraph("Features 3 stacked LSTM layers (128, 64, and 32 units) with dropout (0.2). The gated architecture (forget gate, input gate, output gate) allows LSTMs to retain memory over long-range dependencies, making it optimal for longer forecast horizons (3D, 7D).", body_style))
    
    story.append(Paragraph("Transformer (Time-Series Attention)", h2_style))
    story.append(Paragraph("Implements 2 Transformer encoder blocks featuring Multi-Head Self-Attention (4 attention heads, key dimension = 64) and feed-forward networks (128 units). Self-attention allows direct modeling of temporal associations across any distance in the 60-day window, though highly data-hungry.", body_style))
    
    # --- 4. Validation Strategy ---
    story.append(Paragraph("4. Validation & Evaluation Strategy", h1_style))
    val_text = ("We implement a chronological split (80% train, 20% test) with no shuffling to preserve time ordering. "
                "For validation, 15% of the training partition is held out chronologically. "
                "Random seeds are locked at 42 to ensure deterministic model initialization.")
    story.append(Paragraph(val_text, body_style))
    
    story.append(Paragraph("Walk-Forward Expanding Window Validation", h2_style))
    wf_text = ("For advanced evaluation (as demonstrated in <code>src/walk_forward_demo.py</code>), we use an expanding "
               "window validation strategy. The model starts training on 450 days, tests on the next 90 days, "
               "then expands training to 540 days to test the next 90 days, and so on. This walk-forward process "
               "simulates how a production trading system regularly retrains as new daily prices are recorded.")
    story.append(Paragraph(wf_text, body_style))
    
    # --- 5. Model Comparison Results ---
    story.append(Paragraph("5. Model Performance & Comparison", h1_style))
    story.append(Paragraph("Below are the actual performance metrics compiled from the leak-free model evaluation pipeline. Models are evaluated using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE).", body_style))
    
    # Load and render the results table
    if os.path.exists(CSV_PATH):
        try:
            df = pd.read_csv(CSV_PATH)
            # Round numbers for clean display
            df['MAE'] = df['MAE'].round(2)
            df['RMSE'] = df['RMSE'].round(2)
            df['MAPE (%)'] = df['MAPE (%)'].round(2)
            
            # Convert to table format
            table_data = [[col for col in df.columns]]
            for _, row in df.iterrows():
                table_data.append(list(row.values))
                
            results_table = Table(table_data, colWidths=[80, 100, 100, 100, 100])
            results_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a5276')),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE', (0,0), (-1,0), 10),
                ('BOTTOMPADDING', (0,0), (-1,0), 6),
                ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#f2f4f4')),
                ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#bdc3c7')),
                ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
                ('FONTSIZE', (0,1), (-1,-1), 9),
                ('BOTTOMPADDING', (0,1), (-1,-1), 4),
            ]))
            story.append(results_table)
            story.append(Spacer(1, 10))
        except Exception as e:
            story.append(Paragraph(f"Error loading CSV results: {str(e)}", body_style))
    else:
        story.append(Paragraph("<i>Note: Re-train the models to generate the final performance comparison table here.</i>", body_style))
        
    # --- 6. Business Interpretation & Key Findings ---
    story.append(Paragraph("6. Business Interpretation & Key Findings", h1_style))
    findings = (
        "1. <b>1D-CNN efficiency:</b> 1D-CNN converges rapidly and displays robust performance at the short-term (1D) "
        "horizon. Its causal filters effectively act as dynamic technical indicators.<br/>"
        "2. <b>LSTM for Longer Sequences:</b> Stacked LSTMs show superior capability in maintaining context over "
        "3D and 7D horizons. The gating architecture successfully keeps memory of price trends across the 60-day sequences.<br/>"
        "3. <b>Transformer Data-Hungriness:</b> Transformers require massive datasets. With ~5,000 samples, the "
        "self-attention weights struggle to learn generalizable patterns, leading to overfitting and higher error rates.<br/>"
        "4. <b>SimpleRNN limitations:</b> Vanilla SimpleRNN suffers from vanishing gradients over 60 timesteps, "
        "consequently defaulting to high-error predictions."
    )
    story.append(Paragraph(findings, body_style))
    
    # Build Document
    doc.build(story)
    print("CryptoCast Project Report PDF successfully compiled!")

if __name__ == '__main__':
    build_pdf()
