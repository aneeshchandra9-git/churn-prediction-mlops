
"""
Week 4c Orchestrator: Run threshold tuning.
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
        Path('data/X_test.csv'),
        Path('data/y_test.csv'),
        Path('models/logistic_regression_tuned.pkl'),
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
    
    print("\n✅ All prerequisites met. Proceeding with Week 4c...\n")
    return True


def run_week4c():
    """Execute Week 4c threshold tuning."""
    print("=" * 70)
    print("EXECUTING WEEK 4C: THRESHOLD TUNING")
    print("=" * 70)
    
    try:
        from threshold_tuning import main
        main()
        
        print("\n" + "=" * 70)
        print("✅ WEEK 4C EXECUTION COMPLETE")
        print("=" * 70)
        print("\nOutput files created:")
        print("  - week4_artifacts/precision_recall_curve.png")
        print("  - week4_artifacts/roc_curve.png")
        print("  - week4_artifacts/week4_threshold_tuning_summary.json")
        print("\nNext step:")
        print("  - Review MLflow dashboard: http://localhost:5000")
        print("  - WEEK 4 IS COMPLETE! Ready for Weeks 5-6 (Deployment)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR during execution: {str(e)}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("\n🚀 " * 20)
    print("WEEK 4C: THRESHOLD TUNING")
    print("🚀 " * 20 + "\n")
    
    if not check_prerequisites():
        sys.exit(1)
    
    success = run_week4c()
    sys.exit(0 if success else 1)