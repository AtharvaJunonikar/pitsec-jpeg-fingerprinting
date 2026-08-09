# Quick Reference Card

## First-Time Setup (5 minutes)

### macOS / Linux
```bash
git clone <repository-url>
cd pitsec-jpeg-fingerprinting
bash setup.sh
```

### Windows
```cmd
git clone <repository-url>
cd pitsec-jpeg-fingerprinting
setup.bat
```

---

## Daily Workflow

### Before you start
```bash
# Activate virtual environment
source venv/bin/activate        # macOS/Linux
# OR
venv\Scripts\activate.bat       # Windows

# Get latest changes from team
git pull origin main
```

### After making changes
```bash
# See what you changed
git status

# Stage your changes
git add .
# OR add specific files
git add src/bulk_classify.py

# Commit with a clear message
git commit -m "Add YCbCr channel features to compare() function"

# Push to repository
git push origin main

# Deactivate (when done working)
deactivate
```

---

## Common Commands

### Virtual Environment
```bash
# Activate (do this every session)
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate.bat       # Windows

# Deactivate
deactivate

# Check what's installed
pip list

# Update all packages
pip install --upgrade -r requirements.txt

# Add a new package (and save it)
pip install package-name
pip freeze > requirements.txt
git add requirements.txt
git commit -m "Add package-name"
```

### Running Code
```bash
# Test on a single image
python src/test_jpeg_version.py input.jpg

# Run production pipeline
python src/bulk_classify.py

# Test your changes
python -c "from src.bulk_classify import compare; print('✓ Import works')"
```

### Git Basics
```bash
# See current status
git status

# View your changes
git diff src/bulk_classify.py

# See commit history
git log --oneline

# Undo uncommitted changes
git checkout src/bulk_classify.py

# View branches
git branch -a

# Create a new feature branch (optional)
git checkout -b feature/my-feature
```

---

## File Layout (What Goes Where)

```
✓ COMMIT TO GIT:
  ✓ src/                        (your code)
  ✓ requirements.txt            (dependencies)
  ✓ SETUP.md, README.md         (documentation)
  ✓ .gitignore, setup.sh/bat    (config)

✗ DO NOT COMMIT (already in .gitignore):
  ✗ venv/                       (virtual environment)
  ✗ data/alaska_tif/            (raw data)
  ✗ data/compressed/            (generated JPEGs)
  ✗ output/                     (CSV/ARFF files)
  ✗ __pycache__/                (Python cache)
  ✗ *.csv, *.arff               (generated output)
  ✗ .vscode/, .idea/            (IDE settings)
```

---

## When Things Go Wrong

### "ModuleNotFoundError: No module named 'cv2'"
```bash
# Make sure venv is ACTIVE (check for (venv) in prompt)
source venv/bin/activate

# Reinstall packages
pip install -r requirements.txt
```

### "git push rejected"
```bash
# Someone else pushed first. Pull their changes:
git pull origin main

# If there's a conflict, edit the file manually, then:
git add conflicted_file.py
git commit -m "Resolve merge conflict"
git push origin main
```

### "jpeglib not found"
Install build tools first (one-time):
```bash
# macOS
xcode-select --install

# Ubuntu/Debian
sudo apt-get install build-essential python3-dev

# Windows: Use Microsoft C++ Build Tools
```

Then reinstall:
```bash
pip install --force-reinstall jpeglib
```

---

## Collaboration Tips

1. **Pull before you push**
   ```bash
   git pull origin main
   git push origin main
   ```

2. **Commit frequently** with clear messages
   - ✓ `git commit -m "Add YCbCr features to compare()"`
   - ✗ `git commit -m "updates"`

3. **Don't edit the same file** without coordinating
   - Assign different source files to different people
   - Or split responsibilities (one does YCbCr, other does DCT)

4. **Update requirements.txt** if you add packages
   ```bash
   pip freeze > requirements.txt
   git add requirements.txt
   git commit -m "Add new dependency"
   ```

5. **Test before pushing**
   ```bash
   python src/test_jpeg_version.py  # Quick smoke test
   ```

---

## Project Phases

| Phase | File | Effort | Lead |
|-------|------|--------|------|
| 1 | Environment setup | Low | (Everyone) |
| 2 | YCbCr features | Low | Person A |
| 3 | DCT features | Medium | Person B |
| 4 | Chroma wrinkles | Medium | Person C |
| 5 | Scale to 1000+ images | High | (Everyone) |
| 6 | Build classifier | Medium | Person A |
| 7 | Validation | Medium | Person B |
| 8 | Documentation | Low | (Everyone) |

Assign phases and coordinate via commit messages.

---

## Resources

- **Full setup guide**: `SETUP.md`
- **Project overview**: `README.md`
- **Technical details**: `PITSEC_ProjectGuide.pdf`
- **Git tutorial**: https://git-scm.com/book/en/v2
- **Python venv docs**: https://docs.python.org/3/tutorial/venv.html

---

**Last updated**: August 2026
