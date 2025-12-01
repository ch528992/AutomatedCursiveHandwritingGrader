# Cursive Handwriting Grader

A tool that automatically reads and grades real cursive handwriting from a high quality black and white photo or scan, perfect for teachers!

**You do NOT need to know how to code** — just follow these easy steps.

## How to Run It on Windows

### Step 1: Download This Project
1. Click the green **Code** button → **Download ZIP**
2. Save it anywhere (Desktop is easiest)
3. Right-click the ZIP → **Extract All** → pick a folder
   
### Step 2: Install Python 
1. Go to: https://www.python.org/downloads/
2. Click the big yellow **Download Python** button
3. Run the installer
   - **Important:** Check the box **“Add Python to PATH”**
   - Then click **Install Now**
4. Close the installer when it’s done

### Step 3: Install the Required Tools (Only Once)
1. Open the folder you just extracted 
2. Hold **Shift** and **right-click** inside the folder (on empty space)
3. Choose **“Open PowerShell window here”** or **“Open command window here”**  
   → A black (or blue) window will appear
4. Copy and paste these lines **one at a time** (press Enter after each):

```bash
pip install opencv-python
pip install tensorflow
pip install numpy
pip install pillow
pip install matplotlib
```

# Run Program
```
python CursiveGraderGUI.py
```

## Yeah
*You may need to google how to install any package that's missing*
