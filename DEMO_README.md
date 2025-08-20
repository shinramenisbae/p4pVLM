# P4P Integrated Emotion Recognition Demo Pipeline

This demo showcases the complete integrated emotion recognition system that combines:
- **Passive Sensor (Biosignal)**: PPG data analysis using CNN for valence/arousal prediction
- **Active Sensor (Visual)**: Facial expression analysis using emotion classification
- **Late Fusion Module**: Combines both predictions using weighted averaging (60% visual, 40% biosignal)

## 🎯 Demo Overview

The demo processes:
1. **CSV file** (`tester.csv`) with 43 rows of PPG data (86 seconds of recording)
2. **Video file** (`visual_data_test.mp4`) with 87 seconds of facial expressions
3. **Fuses predictions** from both modalities in real-time
4. **Outputs JSON files** with all predictions and fusion results

## 📁 File Structure

```
p4pVLM/
├── demo_pipeline.py          # Main demo pipeline
├── run_demo.py              # User-friendly runner script
├── demo_requirements.txt    # Python dependencies
├── DEMO_README.md          # This file
├── active/                 # Visual emotion detection
├── passive/                # Biosignal processing
├── late_fusion_module.py   # Fusion logic
└── demo_outputs/           # Generated outputs (created automatically)
    ├── biosignal_predictions.json
    ├── visual_predictions.json
    ├── fused_predictions.json
    └── demo_summary.json
```

## 🚀 Quick Start

### Option 1: Simple Runner (Recommended)
```bash
cd p4pVLM
python run_demo.py
```

### Option 2: Direct Pipeline
```bash
cd p4pVLM
python demo_pipeline.py
```

## 📋 Prerequisites

### Required Files
- `passive/model/network/input-folder/tester.csv` - Your PPG data
- `visual_data_test.mp4` - Your video file
- `passive/model/network/emotion_cnn.pth` - Trained biosignal model

### Python Dependencies
```bash
pip install -r demo_requirements.txt
```

## 🔧 How It Works

### Step 1: Biosignal Processing
- Reads CSV file with PPG data
- Upsamples from 25Hz to 64Hz
- Extracts 140-sample segments
- Runs through CNN model for valence/arousal prediction
- **Output**: `biosignal_predictions.json`

### Step 2: Visual Processing
- Processes video frames every 2 seconds
- Detects faces and emotions using pre-trained model
- Maps discrete emotions to valence/arousal values
- **Output**: `visual_predictions.json`

### Step 3: Late Fusion
- Matches biosignal segments with video frames by timing
- Applies weighted average: 60% visual + 40% biosignal
- Generates final emotion predictions
- **Output**: `fused_predictions.json`

### Step 4: Summary Report
- Analyzes all predictions
- Generates statistics and distributions
- **Output**: `demo_summary.json`

## 📊 Output Format

### Biosignal Predictions
```json
{
  "segment_id": 1,
  "valence": 0,
  "arousal": 1,
  "confidence": 0.85,
  "valence_confidence": 0.82,
  "arousal_confidence": 0.88,
  "timestamp": 1234567890.123,
  "ppg_segment": [2107152, 2104330, ...]
}
```

### Visual Predictions
```json
{
  "frame_id": 60,
  "timestamp": 2.0,
  "face_id": 0,
  "emotion": "happy",
  "confidence": 0.92,
  "valence": 0.8,
  "arousal": 0.6,
  "position": [100, 150, 200, 250],
  "processing_time": 1234567890.123
}
```

### Fused Predictions
```json
{
  "fusion_id": 1,
  "biosignal_segment": 1,
  "visual_frame": 60,
  "timestamp": 1234567890.123,
  "valence": 0.32,
  "arousal": 0.76,
  "discrete_emotion": "happy",
  "fusion_confidence": 0.89,
  "fusion_strategy": "weighted_average",
  "biosignal_contribution": 0.4,
  "visual_contribution": 0.6,
  "metadata": {...}
}
```

## 🎥 Recording the Demo Video

To create a compelling demo video:

1. **Run the pipeline** and record your screen
2. **Show real-time progress** as the system processes data
3. **Highlight key moments**:
   - CSV loading and PPG processing
   - Video frame analysis with emotion detection
   - Fusion process and final predictions
   - Output file generation

4. **Include visual elements**:
   - Progress bars and status indicators
   - Real-time emotion predictions
   - Final results and statistics

## 🔍 Troubleshooting

### Common Issues

**Missing Dependencies**
```bash
pip install -r demo_requirements.txt
```

**Missing Model File**
- Ensure `emotion_cnn.pth` exists in `passive/model/network/`
- Train the model first if needed

**File Path Issues**
- Check that CSV and video files are in correct locations
- Use absolute paths if needed

**Memory Issues**
- Reduce video resolution if processing large files
- Process fewer frames by increasing `frame_interval`

### Debug Mode

For detailed error information, check the console output. The pipeline provides comprehensive logging for each step.

## 📈 Performance Notes

- **Biosignal processing**: Very fast (seconds)
- **Video processing**: Depends on video length and frame rate
- **Fusion**: Near-instantaneous
- **Total time**: Typically 1-5 minutes for 87-second video

## 🎯 Customization

### Adjusting Fusion Weights
```python
pipeline = IntegratedDemoPipeline(
    csv_file_path="your_csv.csv",
    video_file_path="your_video.mp4",
    biosignal_weight=0.3,  # Change from 0.4
    visual_weight=0.7       # Change from 0.6
)
```

### Processing Different File Types
- Modify `process_biosignal_data()` for different CSV formats
- Adjust `process_video_data()` for different video formats
- Update fusion logic for different timing schemes

## 🤝 Contributing

To improve the demo pipeline:
1. Add new fusion strategies
2. Implement real-time visualization
3. Add more input format support
4. Optimize performance
5. Enhance error handling

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review console output for error messages
3. Ensure all prerequisites are met
4. Verify file paths and formats

---

**Happy Demo-ing! 🎉**
