# MLOps Project - Quick Start Guide

## 🚀 Quick Setup (5 minutes)

### Step 1: Create Project Folder
```bash
# Open your terminal/command prompt
mkdir mlops-project
cd mlops-project
```

### Step 2: Initialize Git Repository
```bash
git init
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

### Step 3: Download Files
Download these files from Claude and save them in your `mlops-project` folder:
- `requirements.txt`
- `setup.py`
- `.gitignore`
- `Makefile`
- `README.md`
- `src_init.py` → rename to `src/__init__.py`

Your folder should look like:
```
mlops-project/
├── requirements.txt
├── setup.py
├── .gitignore
├── Makefile
├── README.md
└── src/
    └── __init__.py
```

### Step 4: Set Up Python Environment
```bash
# Create virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Create all project directories
make create-dirs

# Install dependencies
make install
```

### Step 5: Open in Claude Code Desktop

1. **Open Claude Code Desktop** app
2. **Click "File" → "Open Folder"**
3. **Select your `mlops-project` folder**
4. **Click "Open"**

Now Claude Code has full access to your project!

### Step 6: Test Installation
In Claude Code's terminal, run:
```bash
pytest --version
python -c "import mlflow; print(mlflow.__version__)"
make help
```

---

## 📱 How to Use Claude Code + This Chat

### Claude Code is for:
✅ Creating and editing files
✅ Running Python scripts  
✅ Executing tests with `pytest`
✅ Testing API with `python` commands
✅ Git operations (commit, push, pull)
✅ Viewing and debugging code

### This Chat is for:
✅ Asking conceptual questions
✅ Understanding architecture
✅ Getting code snippets to paste
✅ Debugging issues
✅ Explanations on tools
✅ Asking for improvements

### Workflow Example:

**In this Chat:**
```
You: "I don't understand how data pipelines work"
Me: "Here's an explanation... and here's a skeleton..."
```

**In Claude Code:**
```
- Paste the skeleton code
- Create the actual files
- Run the code
- See what happens
```

**Back in this Chat:**
```
You: "I got this error: [paste error]"
Me: "That happens because... fix it by..."
```

---

## 🔧 First Tasks After Setup

### Task 1: Create Directory Structure (5 min)
In Claude Code terminal:
```bash
make create-dirs
```

### Task 2: Verify Installation (5 min)
```bash
make test
```
Should see: `passed` message

### Task 3: Create Basic Config File (10 min)
Ask me in chat:
> "Create a basic config.yaml template"

Then in Claude Code, create `config/config.yaml`

### Task 4: Start First Notebook (15 min)
In Claude Code terminal:
```bash
make run-notebook
```

Then create `notebooks/01_exploration.ipynb` using Jupyter

---

## 📋 Dependencies Included

**Core ML:**
- pandas, numpy, scikit-learn, xgboost, lightgbm

**Experiment Tracking:**
- MLflow, Weights & Biases

**Web Framework:**
- FastAPI, Uvicorn

**Testing:**
- pytest, pytest-cov, hypothesis

**Orchestration:**
- Apache Airflow

**Monitoring:**
- Prometheus client

**Data Quality:**
- Great Expectations

**Code Quality:**
- black, flake8, pylint, mypy

See `requirements.txt` for complete list.

---

## 🐛 Common Issues & Solutions

### Issue: `python: command not found`
**Solution:** Python might not be installed. Download from python.org

### Issue: `No module named mlflow`
**Solution:** Run `make install` (dependencies aren't installed yet)

### Issue: `Permission denied` (macOS/Linux)
**Solution:** Run `chmod +x venv/bin/activate` then try again

### Issue: `ModuleNotFoundError: No module named 'src'`
**Solution:** Make sure `src/__init__.py` exists (not just the folder)

### Issue: Can't open in Claude Code
**Solution:** Make sure you have Claude Code Desktop installed (not just web version)

---

## 📈 Week 1 Tasks

Once setup is complete, follow these in order:

### Days 1-2: EDA
1. Ask me: "Create an EDA notebook template"
2. Paste into Claude Code
3. Run it on sample data
4. Explore the data

### Days 3-4: Data Pipeline
1. Ask me: "Create a data_pipeline.py module"
2. Create file in Claude Code
3. Test it with sample data
4. Debug any issues

### Days 5-7: Baseline Model
1. Ask me: "Create baseline model training script"
2. Run training in Claude Code
3. Save model
4. Evaluate performance

---

## 🔗 Linking Repositories (Optional)

### Create GitHub Repo:
1. Go to github.com
2. Click "New Repository"
3. Name it "mlops-project"
4. Copy the URL

### Link to Local:
```bash
git remote add origin https://github.com/YOUR-USERNAME/mlops-project.git
git branch -M main
git push -u origin main
```

Now all your work syncs with GitHub!

---

## 📊 Progress Checklist

- [ ] Project folder created
- [ ] Git initialized
- [ ] Python virtual environment activated
- [ ] Dependencies installed (`make install` succeeded)
- [ ] All directories created (`make create-dirs` succeeded)
- [ ] Claude Code opened with project folder
- [ ] `make test` passes
- [ ] First notebook created
- [ ] First data pipeline script created
- [ ] First model trained

---

## 🆘 Getting Help

1. **Setup Issues**: Ask me in chat with exact error message
2. **Concept Questions**: Ask me to explain (I'll use Chat)
3. **Code Issues**: Show me the error + code snippet in chat
4. **Git Issues**: Tell me what you tried + error message

---

## 🎯 Next Steps

After setup is complete:

1. **Come back to Chat** and say: "Setup complete, ready for Week 1"
2. I'll guide you through **EDA and data exploration**
3. We'll create the **first data pipeline together**
4. You'll train your **baseline model**

---

## 📚 Useful Commands

```bash
# Create directories
make create-dirs

# Install dependencies
make install

# Run tests
make test

# See code quality
make lint

# Format code
make format

# Start Jupyter
make run-notebook

# Start MLflow UI
make mlflow-ui

# View all commands
make help
```

---

## ✅ You're Ready!

Once you complete setup, you have:
✅ Complete project structure
✅ All dependencies installed
✅ Git ready to use
✅ Claude Code connected to your project
✅ Makefile with 20+ helpful commands
✅ Clear workflow between Chat and Code

**Next: Tell me when setup is complete!**
