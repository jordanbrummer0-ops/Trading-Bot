# Trading Bot - Streamlit Cloud Deployment Guide

## Repository Information
- **Repository**: jordanbrummer0-ops/Trading-Bot
- **Branch**: main
- **Main file**: monitoring_dashboard.py
- **App URL**: .streamlit.app

## Pre-Deployment Setup

### 1. Repository Structure
Ensure your repository has the following structure:
```
Trading-Bot/
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml (template only)
├── src/
│   ├── __init__.py
│   ├── data_fetcher.py
│   ├── visualization.py
│   └── ...
├── monitoring_dashboard.py
├── requirements-streamlit.txt
└── main.py
```

### 2. Requirements File
Use `requirements-streamlit.txt` for Streamlit Cloud deployment (optimized for cloud environment).

### 3. Configuration Files
- `.streamlit/config.toml`: Streamlit app configuration
- `.streamlit/secrets.toml`: Template for secrets (DO NOT commit actual secrets)

## Streamlit Cloud Deployment Steps

### 1. Connect Repository
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select repository: `jordanbrummer0-ops/Trading-Bot`
5. Branch: `main`
6. Main file path: `monitoring_dashboard.py`

### 2. Configure Secrets
In Streamlit Cloud app settings, add secrets:
```toml
[api_keys]
alpaca_api_key = "your_actual_api_key"
alpaca_secret_key = "your_actual_secret_key"

[trading]
paper_trading = true
initial_capital = 10000
```

### 3. Advanced Settings
- Python version: 3.9+
- Requirements file: `requirements-streamlit.txt`

## Important Notes

### Cloud Limitations
- No persistent file storage (use cloud databases)
- Limited compute resources
- No real-time trading execution (dashboard only)
- Session state resets on app restart

### Security Considerations
- Never commit API keys to repository
- Use Streamlit secrets for sensitive data
- Enable paper trading mode for cloud deployment

### Troubleshooting

#### Installer Errors (Non-zero Exit Code):
1. **Try minimal requirements**: Use `requirements-minimal.txt` instead
2. **Version conflicts**: Remove version pins from requirements
3. **Problematic packages**: Remove scipy, aiohttp, seaborn temporarily
4. **Alternative approach**: Deploy with basic functionality first

#### Common Issues:
1. **Import Errors**: Ensure all dependencies are in requirements-streamlit.txt
2. **File Path Issues**: Use relative paths, avoid absolute Windows paths
3. **Memory Limits**: Reduce data loading, use caching
4. **Timeout Issues**: Optimize data fetching and processing
5. **Dependency Conflicts**: Use flexible version ranges (>=) instead of exact versions (==)

#### Solutions:
- Check logs in Streamlit Cloud dashboard
- Use `st.error()` for debugging
- Implement graceful error handling
- Consider caching with `@st.cache_data`
- Start with minimal requirements and add packages incrementally

#### Performance Optimization:
- Use `@st.cache_data` for expensive operations
- Limit historical data range
- Implement lazy loading for charts
- Use session state efficiently

## Testing Locally
Before deploying:
```bash
pip install -r requirements-streamlit.txt
streamlit run monitoring_dashboard.py
```

## Post-Deployment
1. Test all dashboard features
2. Verify data loading works
3. Check performance metrics
4. Monitor app logs for errors

## Support
For deployment issues:
- Check Streamlit Cloud logs
- Review app settings
- Verify secrets configuration
- Test locally first