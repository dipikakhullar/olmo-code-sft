#!/bin/bash
# Training monitoring script for Accelerate-based OLMo training

echo "🚀 OLMO TRAINING MONITOR 🚀"
echo "=================================="

# Check if training job is running
echo "📊 JOB STATUS:"
if jobs -l | grep -q "run_job_local_accelerate.sh"; then
    echo "✅ Training job is RUNNING"
    jobs -l | grep "run_job_local_accelerate.sh"
else
    echo "❌ Training job is NOT running"
fi

echo ""

# Check log file info
if [ -f "training_accelerate.log" ]; then
    echo "📄 LOG FILE INFO:"
    echo "Size: $(ls -lh training_accelerate.log | awk '{print $5}')"
    echo "Last modified: $(ls -l training_accelerate.log | awk '{print $6, $7, $8}')"
    echo ""
    
    echo "📈 LATEST PROGRESS (last 10 lines):"
    echo "-----------------------------------"
    tail -10 training_accelerate.log
    echo "-----------------------------------"
    echo ""
    
    # Look for key training indicators
    echo "🔍 KEY METRICS:"
    echo "GPU Memory usage:"
    grep -i "gpu mem" training_accelerate.log | tail -3
    echo ""
    
    echo "Training Loss (if available):"
    grep -i "training loss" training_accelerate.log | tail -3
    echo ""
    
    echo "Evaluation Loss (if available):"
    grep -i "eval_loss\|validation loss" training_accelerate.log | tail -3
    echo ""
    
    echo "Errors/Warnings (if any):"
    grep -i "error\|warning\|failed\|exception" training_accelerate.log | tail -5
    
else
    echo "❌ Log file 'training_accelerate.log' not found"
fi

echo ""
echo "💡 MONITORING COMMANDS:"
echo "  Watch live log: tail -f training_accelerate.log"
echo "  Check GPU usage: nvidia-smi"
echo "  Kill training: kill $(jobs -l | grep run_job_local_accelerate.sh | awk '{print $2}' 2>/dev/null)"
echo "  Re-run monitor: bash monitor_training.sh" 