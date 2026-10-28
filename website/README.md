# X-UAD website

Results page and live demo for X-UAD. The trained model (best single checkpoint,
94.58% test accuracy) runs in the browser, so no Python packages are needed and no
internet connection is needed once the page is open.

## Run it

Double-click `start_website.bat`. It opens http://localhost:8000 in your browser.
Keep the black window open while presenting.

Without the batch file:

```
cd website
python -m http.server 8000
```

then open http://localhost:8000.

Opening `index.html` directly (double-clicking it) shows the results but not the
live demo: browsers block loading the model from a `file://` page.

## What is in the folder

| Path | Contents |
|---|---|
| `index.html` | The page |
| `model/xuad.onnx` | `results_region_pooled_detached/xuad_P1_regionpool_detached.pt` exported to ONNX |
| `model/haarcascade_frontalface_default.xml` | OpenCV's frontal-face detector, the same one `xuad/data/faces.py` uses |
| `lib/` | onnxruntime-web 1.17.3 and OpenCV.js 4.10 |
| `samples/` | 16 test-split images (2 per class), exported with `demo/export_samples.py --n 2` |
| `img/` | Figures from `results_final/figures/` and `paper/figures/` |

## How faithful the demo is

- The ONNX model's outputs match the PyTorch checkpoint to within 2e-6 on real test images.
- 128 × 128 images (the benchmark format) are centre-cropped to 112 × 112, the processing
  used to compute the reported accuracy. Other images go through Haar face detection and
  cropping, as in `demo/run_demo.py`.
- On test uploads the explanation text is word-for-word identical to `demo/run_demo.py`.

## Putting it online

The folder is a plain static site, so any static host works: upload the whole
`website` folder to Netlify Drop, GitHub Pages or similar. All files are under 11 MB.
