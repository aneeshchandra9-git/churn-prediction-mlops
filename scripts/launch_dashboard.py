import webbrowser
import subprocess
import time
import sys
import os

# Fix for Windows Unicode encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'

def launch_dashboard():
    """Launch MLflow UI and automatically open in browser."""
    
    print("Starting MLflow Dashboard...")
    print("=" * 60)
    
    # Start MLflow server
    mlflow_process = subprocess.Popen(
        [sys.executable, "-m", "mlflow", "ui", 
         "--backend-store-uri", "sqlite:///mlflow.db"],
        cwd="."
    )
    
    # Wait for server to start
    time.sleep(3)
    
    # Open browser automatically
    url = "http://localhost:5000"
    print(f"Opening dashboard at {url}")
    webbrowser.open(url)
    
    print("=" * 60)
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        mlflow_process.wait()
    except KeyboardInterrupt:
        print("\nStopping MLflow server...")
        mlflow_process.terminate()
        mlflow_process.wait()

if __name__ == "__main__":
    launch_dashboard()