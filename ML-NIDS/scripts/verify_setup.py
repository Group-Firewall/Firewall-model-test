"""
Setup Verification Script
Verifies that all components are properly installed and can be imported
"""

import sys
import os

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_path = os.path.join(parent_dir, 'src')
sys.path.insert(0, src_path)

def verify_imports():
    """Verify all modules can be imported"""
    print("="*60)
    print("Verifying NIDS System Setup")
    print("="*60)
    
    errors = []
    
    # Test imports
    modules_to_test = [
        ('data_preprocessing', 'DataPreprocessor'),
        ('supervised_models', 'SupervisedModelTrainer'),
        ('unsupervised_models', 'UnsupervisedModelTrainer'),
        ('signature_detection', 'SignatureDetector'),
        ('hybrid_detection', 'HybridDetector'),
        ('online_learning', 'OnlineLearner'),
        ('evaluation', 'NIDSEvaluator'),
        ('alert_system', 'AlertSystem')
    ]
    
    print("\n1. Testing Module Imports:")
    for module_name, class_name in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"   ✓ {module_name}.{class_name}")
        except Exception as e:
            print(f"   ✗ {module_name}.{class_name} - Error: {e}")
            errors.append(f"{module_name}.{class_name}: {e}")
    
    # Test dependencies
    print("\n2. Testing Dependencies:")
    dependencies = [
        'pandas', 'numpy', 'sklearn', 'xgboost', 
        'matplotlib', 'seaborn', 'plotly', 'streamlit',
        'joblib', 'psutil'
    ]
    
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"   ✓ {dep}")
        except ImportError:
            print(f"   ✗ {dep} - Not installed")
            errors.append(f"{dep} not installed")
    
    # Check optional dependencies
    print("\n3. Testing Optional Dependencies:")
    optional_deps = [
        ('tensorflow', 'TensorFlow (for Autoencoder)'),
        ('river', 'River (for advanced online learning)')
    ]
    
    for dep, desc in optional_deps:
        try:
            __import__(dep)
            print(f"   ✓ {dep} - {desc}")
        except ImportError:
            print(f"   ⚠ {dep} - {desc} (optional, will use fallback)")
    
    # Check directory structure
    print("\n4. Checking Directory Structure:")
    required_dirs = ['src', 'scripts', 'dashboard', 'models', 'results']
    for dir_name in required_dirs:
        dir_path = os.path.join(parent_dir, dir_name)
        if os.path.exists(dir_path):
            print(f"   ✓ {dir_name}/")
        else:
            print(f"   ✗ {dir_name}/ - Missing")
            errors.append(f"Directory {dir_name} missing")
    
    # Summary
    print("\n" + "="*60)
    if errors:
        print("⚠️  Setup Verification: FAILED")
        print(f"\nFound {len(errors)} issue(s):")
        for error in errors:
            print(f"  - {error}")
        print("\nPlease fix the issues above before proceeding.")
    else:
        print("✅ Setup Verification: SUCCESS")
        print("\nAll components are properly installed and ready to use!")
        print("\nNext steps:")
        print("  1. Run: python scripts/train_models.py")
        print("  2. Run: python scripts/evaluate_system.py")
        print("  3. Run: streamlit run dashboard/app.py")
    print("="*60)
    
    return len(errors) == 0


if __name__ == '__main__':
    success = verify_imports()
    sys.exit(0 if success else 1)

