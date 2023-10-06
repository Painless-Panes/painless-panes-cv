# Painless Panes Computer Vision API

Implements the computer vision API used by the [Painless
Panes](https://github.com/Painless-Panes/painless-panes) app.

The window detection model used by this app can be reconstructed and/or improved using
the scripts provided in the [Painless Panes
Model](https://github.com/Painless-Panes/painless-panes-model) repository.

## Features / Usage

Provides two routes:

1. A GET route for pinging the server to test that it is running.
2. A POST route that receives a photograph of a window and returns the width and height
as headers along with an annotated image (showing the detected window and the measured
frame).

## Known Bugs / Defects

1. The window detection model works quite well for its size, but could be improved by
using a larger model with more training data. See instructions in the [Painless Panes
Model](https://github.com/Painless-Panes/painless-panes-model) repository to update this model.
2. The window corner-finding algorithm (`painless_panes/cv.py#find_window_corners`),
which finds line intersections without overhang, works great some of the time, but is unreliable and probably a dead-end in the long-term.
3. The current algorithm does nothing to correct for perspective. Some functions to
correct for perspective were implemented, but they ended up making the measurements
*less* accurate, rather than more. See `painless_panes/_archive.py`.

Some resources that might be helpful for developing a better window corner-finding algorithm:

 - I believe the ideal approach is to use some form of [contour
 detection](https://learnopencv.com/contour-detection-using-opencv-python-c/) with a
 tree-hierarchy retrieval method (`RETR_TREE`), selecting for rectangular contours.
 The strategy would be as follows:
     1. Find all innermost rectangles (rectangles without children in the tree
     hierarchy) inside the bounding box.
     2. The outermost edges of that set of rectangles should be the inner edge of the frame.
     3. The smallest parent containing all of those rectangles should be the outer edge of the frame.
 - This
 [repo](https://github.com/AlaaAnani/Hough-Rectangle-and-Circle-Detection/tree/main) finds rectangles using a Hough transform, which may be useful. One challenge is that these will not be perfect rectangles, due to perspective skewing.

## Running Locally

### Software Needed

You will need the Python package manager [Poetry](https://python-poetry.org/) installed.
It is similar to `npm`.

### Instructions

1. Run `poetry install`
2. Run `flask --app painless_panes/_api.py run --debug --host 0.0.0.0`

## Deployment

### Accounts Needed for Deployment

To deploy this API, you will need a [Heroku account](https://www.heroku.com/pricing) ($5+).

### Heroku Set-up

For first-time deployment, you need to start with the following:
1. Install the heroku CLI
```
npm install -g heroku
```
2. Log in
```
heroku login
```
3. Create the app
```
heroku create painless-panes-cv
```

### Heroku Configuration

You will need to set the following buildpacks:
```
heroku buildpacks:set heroku/python
heroku buildpacks:add --index 1 heroku-community/apt
```

The `apt` buildpack is necessary for installing the `libgl1` dependency (see `Aptfile`).


### Instructions

Once the app is created and configured, you will use the following to deploy the current version:
```
git push heroku main
```
