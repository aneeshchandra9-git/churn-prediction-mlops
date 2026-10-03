"""
Week 4 Orchestrator: Run all Week 4a hyperparameter tuning tasks.
==================================================================

Execution:
  python scripts/run_week4.py

This script:
1. Checks if required data files exist (from Week 3)
2. Runs model_optimization.py
3. Handles errors gracefully
4. Provides summary output
"""

import sys
import os
from pathlib import Path

# Add src to Python path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def check_prerequisites():
    """Verify Week 3 data exists before running Week 4."""
    print("=" * 70)
    print("CHECKING PREREQUISITES")
    print("=" * 70)
    
    required_files = [
        Path('data/X_train_balanced.csv'),
        Path('data/y_train_balanced.csv'),
        Path('data/X_test.csv'),
        Path('data/y_test.csv'),
        Path('models/logistic_regression_model.pkl'),
    ]
    
    missing_files = []
    for file_path in required_files:
        if file_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} (MISSING)")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n❌ ERROR: {len(missing_files)} required file(s) not found.")
        print("\nMake sure Week 3 completed successfully:")
        print("  - Run Week 3 and generate balanced training data")
        print("  - Save all models to models/ folder")
        print("\nThen retry Week 4.")
        return False
    
    print("\n✅ All prerequisites met. Proceeding with Week 4a...\n")
    return True


def run_week4():
    """Execute Week 4a hyperparameter tuning."""
    print("=" * 70)
    print("EXECUTING WEEK 4A: HYPERPARAMETER TUNING")
    print("=" * 70)
    
    try:
        # Import the optimization module
        from model_optimization import main
        
        # Run the pipeline
        main()
        
        print("\n" + "=" * 70)
        print("✅ WEEK 4A EXECUTION COMPLETE")
        print("=" * 70)
        print("\nOutput files created:")
        print("  - models/logistic_regression_tuned.pkl (best tuned model)")
        print("  - week4_artifacts/week4_hyperparameter_tuning_summary.json")
        print("\nNext step:")
        print("  - Review MLflow dashboard: http://localhost:5000")
        print("  - Run Week 4b: Feature Engineering")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR during execution: {str(e)}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("\n🚀 " * 20)
    print("WEEK 4: MODEL OPTIMIZATION")
    print("🚀 " * 20 + "\n")
    
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    # Run Week 4a
    success = run_week4()
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)