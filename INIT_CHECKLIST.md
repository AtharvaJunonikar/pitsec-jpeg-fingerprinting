# PITSEC Project Initialization Checklist

**Use this checklist the first time you set up the repository.**

## Step 1: Create the Git Repository

- [ ] Create a new repository on GitHub / GitLab / Gitea (or your Git host)
  - Repository name: `pitsec-jpeg-fingerprinting`
  - Description: "JPEG library fingerprinting via re-compression artifacts"
  - Initialize with: `.gitignore` (Python), `README.md`, MIT license (optional)

- [ ] Clone the repository locally
  ```bash
  git clone <repository-url>
  cd pitsec-jpeg-fingerprinting
  ```

- [ ] Copy the following files into the repository root (if not already created by your Git host):
  - `.gitignore` — Python/project exclusions
  - `.python-version` — Python 3.10 (for pyenv users)
  - `requirements.txt` — All dependencies
  - `setup.sh` — Linux/macOS automated setup
  - `setup.bat` — Windows automated setup
  - `SETUP.md` — Manual setup instructions
  - `README.md` — Project overview
  - `QUICKSTART.md` — Quick reference card
  - `PITSEC_ProjectGuide.pdf` — Technical guide (copy from your original)

- [ ] Commit and push the initialization
  ```bash
  git add .
  git commit -m "Initial repository structure with setup guides"
  git push origin main
  ```

## Step 2: Invite Team Members

- [ ] Share the repository URL with each team member
- [ ] Grant repository access (if private repo)
- [ ] Have each person follow the setup steps:
  - **macOS/Linux**: `bash setup.sh`
  - **Windows**: `setup.bat`

## Step 3: Create Project Directories (Local Only)

These folders are **NOT committed** to Git (they're in `.gitignore`).

Each team member creates these locally:

```bash
# Create data and output directories
mkdir -p data/alaska_tif
mkdir -p data/compressed
mkdir -p output
mkdir -p src
```

## Step 4: Add Source Code

- [ ] Create the main scripts in the `src/` folder:
  - `test_jpeg_version.py` — Exploratory prototype
  - `bulk_classify.py` — Production pipeline

  Where do these come from? They're either:
  - Ported from existing code you have
  - Implemented following Section 5 of `PITSEC_ProjectGuide.pdf`

- [ ] Commit the scripts
  ```bash
  git add src/
  git commit -m "Add JPEG fingerprinting baseline scripts"
  git push origin main
  ```

## Step 5: Team Data Sync (Optional but Recommended)

If your team shares the ALASKA dataset or pre-generated JPEG files:

- [ ] Upload a compressed archive (ZIP/TAR) to a shared drive or S3 bucket
- [ ] Add a `DATA_SOURCES.md` file to the repo explaining where to download:
  ```markdown
  # Data Sources

  The ALASKA dataset is large and NOT stored in Git.

  Download from: [link to your shared location]

  Extract to: `data/alaska_tif/`
  ```

- [ ] Commit `DATA_SOURCES.md`
  ```bash
  git add DATA_SOURCES.md
  git commit -m "Add data source instructions"
  git push origin main
  ```

## Step 6: Set Up Branches (Optional but Recommended)

For a cleaner workflow with 3+ people, use feature branches:

```bash
# Create branches for major features
git checkout -b feature/ycbcr-features
git checkout -b feature/dct-domain-features
git checkout -b feature/chroma-wrinkles
git checkout -b feature/classifier

# Push to remote
git push -u origin feature/ycbcr-features
```

Then assign each person a branch. When a feature is ready:
1. Push to the feature branch
2. Create a pull request (PR) on GitHub/GitLab
3. Have a teammate review it
4. Merge into `main`

For now (getting started), this is optional — a single `main` branch is fine.

## Step 7: Create a CONTRIBUTING.md (Optional)

If your team is large or distributed, add guidelines:

```markdown
# Contributing

## Workflow

1. Pull before you push: `git pull origin main`
2. Create a feature branch: `git checkout -b feature/description`
3. Make your changes and test them
4. Commit with a clear message: `git commit -m "Add feature: …"`
5. Push and create a pull request
6. A teammate reviews; you merge

## Code Style

- Use 4-space indentation (Python standard)
- Add docstrings to all functions
- Test on at least 5 images before committing

## Responsibilities

- **Person A**: YCbCr features (Section 5.1)
- **Person B**: DCT-domain features (Section 5.2)
- **Person C**: Chroma wrinkles (Section 5.3)
- **Everyone**: Scaling to 1000+ images (Section 5.4)

```

## Step 8: First Full Run (Together)

Once everything is set up:

- [ ] Each person activates their venv: `source venv/bin/activate`
- [ ] Run the baseline script together (on a single test image): `python src/test_jpeg_version.py`
- [ ] Verify output: check `output/` folder (if it exists) or print statement
- [ ] Celebrate — your environment works!

## Troubleshooting During Initialization

### "Git won't let me clone"
- Check your SSH keys or HTTPS token
- Verify repository URL is correct
- Ensure you have access (check with repo owner)

### "setup.sh won't run"
```bash
# Make it executable first
chmod +x setup.sh
# Then run
bash setup.sh
```

### "jpeglib install fails"
See SETUP.md troubleshooting section — you likely need build tools.

### "Can't import cv2 or numpy after installing"
- Confirm venv is active: `which python` should show `.../venv/bin/python`
- Reinstall: `pip install -r requirements.txt`
- Restart your terminal

### "venv/ shows up in git status"
You forgot `.gitignore`. Copy it to the repo root and commit:
```bash
git add .gitignore
git commit -m "Add .gitignore (was missing)"
```

## After Initialization

Your team is ready! Follow the **Implementation Roadmap** in `PITSEC_ProjectGuide.pdf` Section 4:

| Step | Task | Timeline | Effort |
|------|------|----------|--------|
| 1 | Setup (✓ done) | — | Low |
| 2 | YCbCr features | Week 1 | Low |
| 3 | DCT features | Week 1–2 | Medium |
| 4 | Chroma wrinkles | Week 2 | Medium |
| 5 | Scale to 1000+ images | Week 2–3 | High |
| 6 | Build classifier | Week 3 | Medium |
| 7 | Validation | Week 3–4 | Medium |
| 8 | Documentation | Final | Low |

Use the **daily Git workflow** (see Git workflow diagram in this guide) and commit frequently.

---

**You're all set.** Good luck with the project!
