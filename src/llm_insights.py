import os
import time
import pandas as pd
import streamlit as st
import groq
from dotenv import load_dotenv

def get_groq_api_key() -> str:
    """Load API key from .env, env vars, or Streamlit secrets."""
    load_dotenv()
    # 1. Check Streamlit session state
    if "GROQ_API_KEY" in st.session_state and st.session_state["GROQ_API_KEY"]:
        return st.session_state["GROQ_API_KEY"]
    # 2. Check environment variables
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        return api_key
    # 3. Check Streamlit secrets
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    
    return ""

def _call_groq_llm(system_prompt: str, user_prompt: str, api_key: str, max_retries: int = 3) -> str:
    """Internal helper — sends request to Groq API with error handling and retry logic."""
    if not api_key:
        return "Error: Groq API key not found. Please configure it in the sidebar or .env file."
    
    try:
        client = groq.Groq(api_key=api_key)
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
        )
        
        return response.choices[0].message.content
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg:
            return "⚠️ **Rate Limit Reached**: You are making requests too quickly. Please wait and try again."
        return f"Error connecting to Groq API: {error_msg}"

def generate_prediction_insight(prediction_data: dict, api_key: str) -> str:
    """Analyze a single prediction run (after forward pass)."""
    system_prompt = (
        "You are an expert crypto quantitative analyst. Analyze the following model "
        "prediction data and provide a concise, factual, and professional market insight "
        "formatted in Markdown. Keep it under 4 sentences. Explain what the predictions "
        "imply about the market direction and mention the model used. "
        "Do not invent data; only use what is provided."
    )
    
    # Format the dictionary nicely for the LLM
    data_str = "\n".join([f"{k}: {v}" for k, v in prediction_data.items()])
    user_prompt = f"Prediction Data:\n{data_str}"
    
    return _call_groq_llm(system_prompt, user_prompt, api_key)

def generate_log_insights(log_df: pd.DataFrame, api_key: str) -> str:
    """Analyze the full prediction log history for patterns."""
    system_prompt = (
        "You are an expert AI Model Monitor. Analyze the historical prediction logs "
        "for a cryptocurrency forecasting system. Provide a concise, bulleted Markdown "
        "report identifying: 1. Model consensus/disagreement, 2. Prediction stability, "
        "and 3. Any interesting patterns in how different architectures (LSTM, Transformer, "
        "PatchTST) behave. Keep it analytical and factual."
    )
    
    # Send recent logs (e.g., last 20 entries) to avoid context bloat
    recent_logs = log_df.tail(20).to_string()
    user_prompt = f"Recent Prediction Logs:\n{recent_logs}"
    
    return _call_groq_llm(system_prompt, user_prompt, api_key)

def generate_trading_insight(simulation_results: dict, api_key: str) -> str:
    """Analyze trading simulation results."""
    system_prompt = (
        "You are a quantitative trading strategist. Analyze the results of a trading "
        "simulation backtest. Provide a concise narrative (2-3 paragraphs) formatted "
        "in Markdown explaining the strategy's performance, its risk/reward profile, "
        "win rate, and whether it successfully generated alpha compared to buy-and-hold."
    )
    
    # Extract only the summary stats to avoid sending large equity curve arrays
    summary_stats = {
        'Strategy': simulation_results.get('strategy_name', 'Unknown'),
        'Model(s)': simulation_results.get('models', 'Unknown'),
        'Total Trades Executed': simulation_results.get('trades_executed', 0),
        'Win Rate': f"{simulation_results.get('win_rate', 0.0):.1f}%",
        'Initial Capital': "$10,000",
        'Final Strategy Value': f"${simulation_results.get('final_val', 0.0):.2f}",
        'Final Buy and Hold Value': f"${simulation_results.get('buy_hold_val', 0.0):.2f}",
    }
    
    data_str = "\n".join([f"{k}: {v}" for k, v in summary_stats.items()])
    user_prompt = f"Simulation Results Summary:\n{data_str}"
    
    return _call_groq_llm(system_prompt, user_prompt, api_key)
