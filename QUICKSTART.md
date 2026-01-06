# 🚀 Quick Start Guide

## Complete Beginner's Guide to Running This Project

---

## ⚡ Fastest Way: Google Colab (Recommended)

**Time needed: 5 minutes**

1. **Open the notebook**
   - Go to: https://colab.research.google.com
   - Click `File` → `Upload notebook`
   - Upload `snap_lens_demo.ipynb` from this project

2. **Enable GPU**
   - Click `Runtime` → `Change runtime type`
   - Select `GPU` from Hardware accelerator
   - Click `Save`

3. **Run everything**
   - Click `Runtime` → `Run all`
   - Wait ~5 minutes for setup
   - Upload your selfie when prompted
   - See results!

**That's it!** ✅

---

## 🏃 Alternative: Kaggle

**Time needed: 5 minutes**

1. **Create Kaggle account** (if you don't have one)
   - Go to kaggle.com
   - Sign up (free)

2. **Upload notebook**
   - Click `Code` → `New Notebook`
   - Click `File` → `Upload notebook`
   - Select `snap_lens_demo.ipynb`

3. **Enable GPU**
   - Click `Settings` (right sidebar)
   - Under `Accelerator`, select `GPU T4 x2`
   - Click `Save`

4. **Run all cells**
   - Click `Run all`
   - Follow instructions in notebook

---

## 💻 Local Setup (Advanced)

**Prerequisites:**
- Python 3.8+
- NVIDIA GPU (recommended, not required)
- 10GB free disk space

**Steps:**

```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/snap-genai-lens.git
cd snap-genai-lens

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run demo
python app.py
```

Then open your browser to `http://localhost:7860`

---

## 📱 What You Need

### For Colab/Kaggle (Easiest)
- ✅ Google account
- ✅ Web browser
- ✅ Your selfie image

### For Local Setup
- ✅ Python 3.8 or higher
- ✅ NVIDIA GPU (optional, but much faster)
- ✅ 10GB free disk space
- ✅ Stable internet (for downloading models)

---

## 🎯 Step-by-Step Walkthrough

### Using Colab (Absolute Beginner)

**Step 1: Upload the notebook**
```
1. Go to colab.research.google.com
2. You'll see a dialog - click "Cancel"
3. Click "File" in top menu
4. Click "Upload notebook"
5. Click "Choose File"
6. Select "snap_lens_demo.ipynb" from where you downloaded this project
7. Click "Open"
```

**Step 2: Turn on GPU**
```
1. Click "Runtime" in top menu
2. Click "Change runtime type"
3. Under "Hardware accelerator", select "GPU"
4. Click "Save"
```

**Step 3: Run everything**
```
1. Click "Runtime" in top menu
2. Click "Run all"
3. Wait for cells to execute (takes ~5 minutes first time)
4. When you see "Upload file", click "Choose Files"
5. Select your selfie
6. Wait for generation (~30 seconds)
7. See your styled image!
```

---

## ❓ Common Issues

### "ModuleNotFoundError"
**Problem:** Missing dependencies
**Solution:** Make sure you ran the first cell that installs packages

### "CUDA out of memory"
**Problem:** Image too large or batch too big
**Solution:** Use smaller image (max 1024x1024) or restart runtime

### "No face detected"
**Problem:** Face not visible or too small
**Solution:** Use clear, frontal selfie with face taking up most of image

### "Generation failed"
**Problem:** Model download interrupted
**Solution:** Restart runtime and run again (models will resume download)

---

## 💡 Tips for Best Results

1. **Use good photos:**
   - Clear, well-lit selfie
   - Face taking up 60%+ of image
   - Looking at camera
   - No sunglasses or masks

2. **Optimal settings:**
   - Steps: 20 (good balance)
   - Guidance: 7.5 (default)
   - Seed: 42 (for consistency)

3. **Try different styles:**
   - Start with 'anime' (most impressive)
   - Then try 'cyberpunk', 'sketch'
   - Each takes ~1-2 seconds

---

## 📊 What to Expect

### First Run (5-10 minutes)
- Downloads models (~4GB)
- Installs dependencies
- Loads everything into memory

### Subsequent Runs (<1 minute)
- Models already downloaded
- Just loads into GPU
- Ready to generate!

### Per Generation (1-2 seconds)
- Face detection: 0.1s
- Conditioning: 0.1s
- Diffusion: 1.0s
- Evaluation: 0.2s

---

## 🆘 Need Help?

### Colab Issues
- Check Runtime → View runtime logs
- Make sure GPU is enabled
- Try restarting runtime

### Quality Issues
- Use better input photo
- Increase inference steps (20 → 30)
- Try different styles

### Still Stuck?
- Open an issue on GitHub
- Include error message
- Share what you tried

---

## 🎓 Next Steps

After you get it working:

1. **Try different photos** - See how it handles various faces
2. **Adjust parameters** - Experiment with steps, guidance
3. **Read the code** - Learn how it works
4. **Modify it** - Add your own style or features

---

## 🎯 For Recruiters/Interviewers

**Quick Demo Path:**
1. Open `snap_lens_demo.ipynb` in Colab
2. Enable GPU
3. Run all (~5 min)
4. See full pipeline in action

**Or just watch:**
- Check `outputs/` folder for example results
- See `README.md` for technical details
- Review `INTERVIEW_PREP.md` for discussion points

---

## ✅ Success Checklist

- [ ] Colab notebook opens
- [ ] GPU is enabled
- [ ] All cells run without errors
- [ ] Face detection works on your selfie
- [ ] Generated image appears
- [ ] Metrics are displayed
- [ ] You can download results

If all checked ✅ - you're done! 🎉

---

**Questions?** Check the main README.md or INTERVIEW_PREP.md
