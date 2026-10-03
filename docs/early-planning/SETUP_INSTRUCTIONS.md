# 🚀 MLOps Project - Complete Setup Instructions

## Overview
This guide will take you through setting up your MLOps project and opening it in Claude Code Desktop in **10 minutes**.

---

## Part 1: Download Files (2 minutes)

### Files You Need to Download
From your Claude chat, download these files:
1. ✅ `requirements.txt`
2. ✅ `setup.py`
3. ✅ `.gitignore`
4. ✅ `Makefile`
5. ✅ `README_SETUP.md`
6. ✅ `setup_project.py`
7. ✅ `src_init.py`

Save all to one folder on your computer.

---

## Part 2: Create Project Folder (1 minute)

### On Windows (Command Prompt)
```cmd
cd Desktop
mkdir mlops-project
cd mlops-project
```

### On macOS/Linux (Terminal)
```bash
cd Desktop
mkdir mlops-project
cd mlops-project
```

---

## Part 3: Move Downloaded Files (1 minute)

1. Locate your downloaded files
2. Move them all into the `mlops-project` folder
3. Rename `src_init.py` to `src/__init__.py` (requires creating `src` folder):
   - Create a folder named `src` inside `mlops-project`
   - Move `src_init.py` into the `src` folder
   - Rename it to `__init__.py`

Your folder should look like:
```
mlops-project/
├── requirements.txt
├── setup.py
├── setup_project.py
├── Makefile
├── .gitignore
├── README_SETUP.md
└── src/
    └── __init__.py
```

---

## Part 4: Set Up Python Environment (3 minutes)

### Step 4a: Open Terminal in Project Folder

**Windows:**
- Right-click inside `mlops-project` folder
- Click "Open in Terminal" (or "Open in PowerShell")

**macOS:**
- Open Terminal app
- Type: `cd ` then drag the `mlops-project` folder into Terminal and press Enter

**Linux:**
- Open Terminal
- `cd ~/Desktop/mlops-project`

### Step 4b: Create Virtual Environment

**All Platforms:**
```bash
python3 -m venv venv
```

Wait 1-2 minutes for it to complete.

### Step 4c: Activate Virtual Environment

**Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
venv\Scripts\activate.bat
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

✅ You should see `(venv)` at the start of your terminal line

### Step 4d: Run Auto-Setup Script

```bash
python setup_project.py
```

This creates all directories and sample files automatically.

### Step 4e: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

⏱️ This takes 2-3 minutes. Grab a coffee! ☕

### Step 4f: Verify Installation

```bash
python -c "import mlflow; print('MLflow version:', mlflow.__version__)"
python -c "import pandas; print('Pandas version:', pandas.__version__)"
pytest --version
```

You should see version numbers. If you see errors, ask me in chat!

---

## Part 5: Open in Claude Code Desktop (2 minutes)

### Step 5a: Launch Claude Code

1. Download and install **Claude Code Desktop** if you haven't already
   - Go to: https://github.com/Anthropic/claude-code-desktop
   - Or search "Claude Code Desktop" online
2. Open the Claude Code Desktop app

### Step 5b: Open Folder

1. Click menu (≡) in top-left
2. Click "Open Folder" or "File" → "Open Folder"
3. Navigate to your `mlops-project` folder
4. Click "Select Folder" or "Open"

### Step 5c: Trust the Folder

If prompted "Do you trust the authors of the files in this folder?", click "Yes, I trust the authors"

✅ **You're now ready to code!**

---

## Part 6: First Test (1 minute)

In Claude Code's terminal (bottom of screen):

```bash
# See all available commands
make help

# Verify everything works
make test

# Should show "passed" ✓
```

If `make test` shows errors, that's fine for now - we'll fix it. The important thing is Python and pytest are working.

---

## Part 7: Verify All Pieces Work

Run these commands one by one in Claude Code terminal:

```bash
# Check project structure
ls -la

# Check Python version
python --version

# Check key libraries
python -c "import pandas, numpy, sklearn, mlflow, fastapi; print('All imports successful!')"

# Check make commands
make help
```

All should work without errors.

---

## Common Setup Issues & Fixes

### Issue: `python: command not found`
**Cause:** Python isn't installed
**Fix:** 
- Download Python from https://www.python.org/
- During installation, **check "Add Python to PATH"**
- Restart terminal/computer
- Try again

### Issue: `venv not found` or permission errors
**Windows Fix:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\venv\Scripts\Activate.ps1
```

**macOS/Linux Fix:**
```bash
chmod +x venv/bin/activate
source venv/bin/activate
```

### Issue: `pip: command not found`
**Fix:**
```bash
python -m pip install --upgrade pip
```

### Issue: `make: command not found` (Windows)
**Fix:** Rename `Makefile` to `makefile` and try again

### Issue: Can't open Claude Code with folder
**Fix:**
1. Close Claude Code completely
2. Open it again
3. Try "File" → "Open Folder" (instead of folder context menu)

### Issue: `ModuleNotFoundError` when running code
**Likely causes:**
- Virtual environment not activated
- Dependencies not installed
- Check: Is `(venv)` showing in terminal?

---

## 🎯 What You Have Now

After setup, you have:
- ✅ Complete project folder structure
- ✅ Python virtual environment isolated and ready
- ✅ All 50+ libraries installed
- ✅ Claude Code connected to your project
- ✅ Makefile with 20+ helpful commands
- ✅ Git initialized and ready
- ✅ Config files for different components
- ✅ Docker setup ready
- ✅ GitHub Actions CI/CD workflow template

---

## 📝 Next: Link with Chat

Once setup is complete:

1. **Tell me in chat:** "Setup complete! Ready for Week 1"
2. **I'll provide:**
   - EDA notebook template
   - Data pipeline skeleton code
   - First baseline model script
3. **You'll:**
   - Paste code into Claude Code
   - Modify for your dataset
   - Run and test it
   - Ask questions in chat

---

## 🔄 Claude Code ↔ Chat Workflow

### In Claude Code:
- ✅ Create files
- ✅ Run Python scripts
- ✅ Execute tests
- ✅ See real results
- ✅ Git operations

### In Chat:
- ✅ Ask concepts
- ✅ Get code templates
- ✅ Debug issues
- ✅ Understand architecture
- ✅ Discuss design

### Example Workflow:

**Chat:** "I need a function to load and validate data"
↓
**Claude Code:** Paste the function → Run it → See errors
↓
**Chat:** Show me the error → "Ah, change line 5 to..."
↓
**Claude Code:** Edit → Test again → Works! ✓

---

## 📚 Useful Commands After Setup

```bash
# See all commands
make help

# Run tests
make test

# Format code nicely
make format

# Check code quality
make lint

# Start Jupyter for notebooks
make run-notebook

# Start MLflow UI (for experiment tracking)
make mlflow-ui

# View Docker status
make docker-build

# Create all directories (if missing)
make create-dirs
```

---

## 🚨 If Something Goes Wrong

1. **Note the exact error message**
2. **Come to chat and paste it:**
   ```
   You: "Setup error - [paste the full error here]"
   ```
3. **Tell me what command caused it**
4. **Tell me your operating system**

I can help fix any issue!

---

## ✅ Setup Checklist

Before moving to Week 1, verify:

- [ ] Project folder created: `mlops-project/`
- [ ] Downloaded 7 files and placed in folder
- [ ] Virtual environment created: `venv/`
- [ ] Virtual environment activated: `(venv)` shows in terminal
- [ ] Setup script ran: `python setup_project.py`
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] All tests pass: `make test` (or at least runs without crashing)
- [ ] Claude Code opened with project folder
- [ ] All 4 commands work without errors:
  - [ ] `make help`
  - [ ] `python --version`
  - [ ] `python -c "import mlflow"`
  - [ ] `pytest --version`

---

## 🎉 Congratulations!

You now have:
- Professional project structure
- All tools configured
- Claude Code ready to use
- Clear workflow established

**Next step:** Come back to chat and say:
> "Setup complete! Show me Week 1 tasks"

Then we'll build your first data pipeline together! 🚀
