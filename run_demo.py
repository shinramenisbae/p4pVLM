#!/usr/bin/env python3
"""
Simple runner script for the P4P Integrated Emotion Recognition Demo Pipeline

This script provides a user-friendly way to run the complete demo and handles
common setup issues.
"""

import os
import sys
import subprocess
import importlib.util

def check_dependencies():
    
    required_packages = [
        'cv2', 'torch', 'numpy', 'pandas', 'scipy', 'transformers', 'PIL'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'torch':
                import torch
            elif package == 'numpy':
                import numpy
            elif package == 'pandas':
                import pandas
            elif package == 'scipy':
                import scipy
            elif package == 'transformers':
                import transformers
            elif package == 'PIL':
                from PIL import Image
            print(f"  ✅ {package}")
        except ImportError:
            print(f"   {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n Missing packages: {', '.join(missing_packages)}")
        print("💡 Install missing packages using:")
        print("   pip install -r demo_requirements.txt")
        return False
    
    print("✅ All dependencies are available!")
    return True

def check_files():
    """Check if required input files exist"""
    print("\n📁 Checking input files...")
    
    required_files = [
        "passive/model/network/input-folder/tester.csv",
        "visual_data_test.mp4"
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} - MISSING")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n❌ Missing files: {', '.join(missing_files)}")
        print("💡 Please ensure all required files are in the correct locations:")
        print("   - CSV file should be in: passive/model/network/input-folder/tester.csv")
        print("   - Video file should be in: visual_data_test.mp4")
        return False
    
    print("✅ All input files are available!")
    return True

def check_models():
    """Check if trained models exist"""
    print("\n🧠 Checking trained models...")
    
    model_files = [
        "passive/model/network/emotion_cnn.pth"
    ]
    
    missing_models = []
    
    for model_path in model_files:
        if os.path.exists(model_path):
            print(f"  ✅ {model_path}")
        else:
            print(f"  ❌ {model_path} - MISSING")
            missing_models.append(model_path)
    
    if missing_models:
        print(f"\n❌ Missing model files: {', '.join(missing_models)}")
        print("💡 Please ensure trained models are available:")
        print("   - Biosignal CNN model should be in: passive/model/network/emotion_cnn.pth")
        print("   - You may need to train the model first using the existing training scripts")
        return False
    
    print("✅ All model files are available!")
    return True

def install_requirements():
    """Install requirements if needed"""
    print("\n📦 Installing requirements...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "demo_requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements")
        print("💡 Try installing manually:")
        print("   pip install -r demo_requirements.txt")
        return False

def run_demo():
    """Run the demo pipeline"""
    print("\n🚀 Starting the demo pipeline...")
    
    try:
        # Import and run the demo
        from demo_pipeline import IntegratedDemoPipeline
        
        # Initialize pipeline
        pipeline = IntegratedDemoPipeline(
            csv_file_path="passive/model/network/input-folder/tester.csv",
            video_file_path="visual_data_test.mp4",
            biosignal_weight=0.4,
            visual_weight=0.6
        )
        
        # Run the demo
        results = pipeline.run_demo()
        
        print("\n🎉 Demo completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    print("🎯 P4P Integrated Emotion Recognition Demo Pipeline")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        print("\n💡 Would you like to install missing dependencies? (y/n)")
        response = input().lower().strip()
        if response == 'y':
            if not install_requirements():
                return
            # Re-check dependencies
            if not check_dependencies():
                return
        else:
            print("❌ Cannot proceed without required dependencies")
            return
    
    # Check files
    if not check_files():
        print("\n❌ Cannot proceed without required input files")
        return
    
    # Check models
    if not check_models():
        print("\n❌ Cannot proceed without trained models")
        return
    
    # Run the demo
    if run_demo():
        print("\n🎊 Demo completed successfully!")
        print("📁 Check the 'demo_outputs' folder for all results!")
    else:
        print("\n💥 Demo failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
