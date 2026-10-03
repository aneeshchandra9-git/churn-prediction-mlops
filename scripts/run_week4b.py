"""
Week 4b Orchestrator: Run feature engineering.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def check_prerequisites():
    """Verify Week 4a completed successfully."""
    print("=" * 70)
    print("CHECKING PREREQUISITES")
    print("=" * 70)
    
    required_files = [
        Path('data/X_train_balanced.csv'),
        Path('data/y_train_balanced.csv'),
        Path('data/X_test.csv'),
        Path('data/y_test.csv'),
        Path('week4_artifacts/week4_hyperparameter_tuning_summary.json'),
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
        print("Make sure Week 4a completed successfully.")
        return False
    
    print("\n✅ All prerequisites met. Proceeding with Week 4b...\n")
    return True


def run_week4b():
    """Execute Week 4b feature engineering."""
    print("=" * 70)
    print("EXECUTING WEEK 4B: FEATURE ENGINEERING")
    print("=" * 70)
    
    try:
        from feature_engineering import main
        main()
        
        print("\n" + "=" * 70)
        print("✅ WEEK 4B EXECUTION COMPLETE")
        print("=" * 70)
        print("\nOutput files created:")
        print("  - models/logistic_regression_engineered.pkl")
        print("  - week4_artifacts/week4_feature_engineering_summary.json")
        print("\nNext step:")
        print("  - Review MLflow dashboard: http://localhost:5000")
        print("  - Run Week 4c: Threshold Tuning (optional)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR during execution: {str(e)}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("\n🚀 " * 20)
    print("WEEK 4B: FEATURE ENGINEERING")
    print("🚀 " * 20 + "\n")
    
    if not check_prerequisites():
        sys.exit(1)
    
    success = run_week4b()
    sys.exit(0 if success else 1)