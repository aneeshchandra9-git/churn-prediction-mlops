# MLOps Project - Quick Reference Card

## 🎯 Daily Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                  START YOUR DAY                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Check Chat for today's task                            │
│     └─ Opens in browser                                    │
│                                                             │
│  2. Open Claude Code (if closed)                           │
│     └─ Opens separate app                                  │
│                                                             │
│  3. Activate venv in Claude Code terminal:                 │
│     • Windows: .\venv\Scripts\activate                     │
│     • Mac/Linux: source venv/bin/activate                  │
│                                                             │
│  4. Read code snippet from Chat                            │
│     └─ Copy the code                                       │
│                                                             │
│  5. Create/edit file in Claude Code                        │
│     └─ File → New File                                     │
│                                                             │
│  6. Paste code and modify                                  │
│     └─ Save the file                                       │
│                                                             │
│  7. Test in Claude Code terminal                           │
│     └─ python filename.py OR pytest                        │
│                                                             │
│  8. If error: show error in Chat                           │
│     └─ I explain and suggest fix                           │
│                                                             │
│  9. Ask questions anytime!                                 │
│     └─ Switch to Chat tab                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 💻 Essential Terminal Commands

### Setup (First Time)
```bash
python -m venv venv              # Create environment
source venv/bin/activate         # Activate (Mac/Linux)
# OR
.\venv\Scripts\activate          # Activate (Windows)

pip install -r requirements.txt  # Install all libraries
make create-dirs                 # Create all folders
```

### Every Time You Start
```bash
source venv/bin/activate         # Make sure venv is active
# See: (venv) in terminal
```

### Daily Commands
```bash
make test                        # Run all tests
make lint                        # Check code quality
make format                      # Auto-fix formatting
pytest tests/                    # Run specific tests
python src/data_pipeline.py     # Run a script
```

### Jupyter & Notebooks
```bash
make run-notebook               # Start Jupyter Lab
# Then open notebooks/
jupyter notebook                # Alternative command
```

### Experiments & Tracking
```bash
make mlflow-ui                  # Start MLflow tracking
# Open http://localhost:5000 in browser
```

### API Development
```bash
make run-api                    # Start FastAPI
# Open http://localhost:8000 in browser
```

### Git
```bash
git status                      # See changes
git add .                       # Stage all files
git commit -m "message"         # Commit changes
git push                        # Push to GitHub
git log --oneline               # See history
```

### Cleanup
```bash
make clean                      # Remove cache files
make clean-cache               # Remove __pycache__
```

---

## 📁 Project Structure at a Glance

```
mlops-project/
│
├── 📓 notebooks/              ← Jupyter experiments
│   ├── 01_exploration.ipynb
│   ├── 02_features.ipynb
│   └── 03_models.ipynb
│
├── 🐍 src/                    ← Main code (MOST IMPORTANT)
│   ├── __init__.py
│   ├── data_pipeline.py       ← Data loading, cleaning
│   ├── feature_engineering.py ← Create features
│   ├── model_training.py      ← Train model
│   ├── model_inference.py     ← Make predictions
│   └── monitoring.py          ← Track performance
│
├── 🧪 tests/                  ← Unit tests
│   ├── test_data_pipeline.py
│   ├── test_features.py
│   └── test_model.py
│
├── 📊 data/                   ← Data files
│   ├── raw/                   ← Original (never edit!)
│   ├── processed/             ← Cleaned data
│   └── external/              ← 3rd party data
│
├── 🤖 models/                 ← Trained models
│   ├── model_v1.pkl
│   └── latest_model.pkl
│
├── ⚙️ config/                  ← Configuration files
│   ├── config.yaml
│   └── logging.yaml
│
├── 🐳 docker/                 ← Containerization
│   ├── Dockerfile
│   └── requirements.txt
│
├── 🚀 deployment/             ← API & serving
│   ├── api.py                 ← FastAPI app
│   └── docker-compose.yml
│
├── 🔄 dags/                   ← Airflow workflows
│   └── ml_pipeline_dag.py
│
├── 📦 config files            ← Root level
│   ├── requirements.txt        ← All dependencies
│   ├── setup.py
│   ├── Makefile               ← Commands
│   ├── .gitignore
│   └── README.md
│
└── 🔧 venv/                   ← Virtual environment (auto-created)
```

**Remember:** Most coding happens in `src/` folder!

---

## 🔗 Chat ↔ Claude Code Quick Map

| I Need... | Chat | Code |
|-----------|------|------|
| Explain concept | ✅ | - |
| Code template | ✅ | - |
| Fix error | ✅ | - |
| Create file | - | ✅ |
| Run script | - | ✅ |
| Test code | - | ✅ |
| Debug | Both | Both |
| Save to Git | - | ✅ |

---

## 📊 Key Files to Know

| File | Purpose | When to Touch |
|------|---------|---------------|
| `requirements.txt` | Dependencies | When adding new libraries |
| `src/data_pipeline.py` | Load/clean data | Modify for your dataset |
| `src/model_training.py` | Train model | When trying new models |
| `tests/test_*.py` | Unit tests | After creating functions |
| `config/config.yaml` | Settings | Adjust hyperparameters |
| `deployment/api.py` | API endpoint | Week 4 |
| `dags/ml_pipeline_dag.py` | Workflow | Week 4 |
| `Makefile` | Commands | Just read it |
| `notebooks/*.ipynb` | Experiments | Early exploration |

---

## 🚨 Common Errors & Quick Fixes

| Error | Solution |
|-------|----------|
| `No module named 'X'` | Run: `pip install -r requirements.txt` |
| `(venv) not showing` | Run: `source venv/bin/activate` or `..\venv\Scripts\activate` |
| `Permission denied` | Run: `chmod +x venv/bin/activate` then activate |
| `Tests fail` | Check: Are imports correct? File in right folder? |
| `Can't import src` | Check: Is `src/__init__.py` present? |
| `Make command not found` | Try: `python -m make` or check Makefile exists |
| `Port already in use` | Kill process: `lsof -i :8000` (Mac/Linux) |
| `Git not found` | Download from: https://git-scm.com/ |

---

## 📈 Weekly Milestones

### Week 1: Data ✅
```bash
✓ EDA complete
✓ data_pipeline.py working
✓ Baseline model trained
```

### Week 2: Models 🤖
```bash
✓ 5+ models tested
✓ MLflow tracking setup
✓ Best model selected
```

### Week 3: Testing & Deployment 🧪
```bash
✓ 80%+ test coverage
✓ Docker image builds
✓ CI/CD pipeline working
```

### Week 4: Production 🚀
```bash
✓ Airflow DAG running
✓ API serving predictions
✓ Monitoring active
```

---

## 🎓 Learning Resources (Bookmark These!)

```
MLflow:      https://mlflow.org/docs/
Airflow:     https://airflow.apache.org/docs/
FastAPI:     https://fastapi.tiangolo.com/
Pandas:      https://pandas.pydata.org/docs/
Scikit-Learn: https://scikit-learn.org/stable/documentation.html
Pytest:      https://docs.pytest.org/
Docker:      https://docs.docker.com/
Git:         https://git-scm.com/doc
```

---

## 🔑 Important Reminders

1. **Always activate venv first**
   - Windows: `.\venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`

2. **Never edit `data/raw/`**
   - Original data stays original
   - Create processed version

3. **Test frequently**
   - Run: `make test` every time you change code
   - Better to catch bugs early

4. **Commit to Git regularly**
   - `git add .` → `git commit -m "message"` → `git push`
   - Creates backup and progress record

5. **Ask in Chat**
   - Stuck? Ask me immediately
   - Conceptual issue? Ask
   - Error you don't understand? Paste it

6. **Keep it simple Week 1**
   - Don't over-engineer
   - Get it working first
   - Optimize later

---

## ⌨️ Keyboard Shortcuts (Claude Code)

```
Ctrl/Cmd + S       Save file
Ctrl/Cmd + K       Open command palette
Ctrl/Cmd + /       Comment/uncomment
Alt + Up/Down      Move line
Ctrl/Cmd + X       Delete line
Ctrl/Cmd + D       Duplicate line
Ctrl/Cmd + Shift + P  Format document
```

---

## 📱 Quick Debug Checklist

When code breaks:
- [ ] Check: Does file exist where expected?
- [ ] Check: Is venv activated?
- [ ] Check: Are imports at top correct?
- [ ] Check: Is data file in right folder?
- [ ] Check: Did you save the file?
- [ ] Check: Python version (python --version)
- [ ] Run: `make clean` then try again
- [ ] Still stuck? Paste error in Chat

---

## ✅ Pre-Week-1 Checklist

```
Before starting Week 1, verify:

Terminal:
  ✓ cd mlops-project (no errors)
  ✓ (venv) shows in terminal
  ✓ make help (shows all commands)
  ✓ make test (runs without crashing)

Claude Code:
  ✓ Project folder is open
  ✓ Terminal at bottom shows (venv)
  ✓ Can create new file
  ✓ Can run Python: python --version

Chat:
  ✓ Can ask questions
  ✓ Can receive code snippets
  ✓ Can paste errors for debugging
```

---

## 🆘 Still Stuck?

1. Take screenshot of error
2. Come to Chat
3. Share:
   - What you were doing
   - Exact error message
   - What command caused it
4. I'll help fix it

**Remember:** No question is too basic. We're learning together! 🚀
