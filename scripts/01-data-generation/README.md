#### 1. Download images

Run the image download scrip:
```
python 01_download.py
```

#### 2. Label images

Look through the downloaded images in `images/` and pick which ones you want to annotate
and use as training data.
Put the remaining images in `unused-images/`.
These can be used for testing.

Run `labelImg`,
```
labelImg
```
Open the `images/` directory, set the output format to `YOLO` and set the output dir to `labels/`.
Then use the GUI to draw bounding boxes around each image.

> Annoyingly, with the current PyPI version of `labelImg`, you will need to fix the
> following lines in order for it to work with recent Python versions
> (see [here](https://github.com/HumanSignal/labelImg/issues/811#issuecomment-977605722)):
> 
> - .venv/lib/python3.11/site-packages/libs/canvas.py (line 526)
> ```
> p.drawRect(int(left_top.x()), int(left_top.y()), int(rect_width), int(rect_height))
> ```
> - .venv/lib/python3.11/site-packages/libs/canvas.py (lines 530-531)
> ```
> p.drawLine(int(self.prev_point.x()), 0, int(self.prev_point.x()), int(self.pixmap.height()))
> p.drawLine(0, int(self.prev_point.y()), int(self.pixmap.width()), int(self.prev_point.y()))
> ```
> - .venv/lib/python3.11/site-packages/labelImg/labelImg.py (line L965)
> ```
> bar.setValue(int(bar.value() + bar.singleStep() * units))
> ```
>