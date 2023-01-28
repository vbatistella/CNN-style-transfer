# Style Transfer
### Summary
Using feature extraction and an optimizer, we can get the important features from a picture, the style from other picture and mix both. As a result we have the first picture with the style from the second.

### Uses
- Creativity.
- Arts.

### Considerations
- The model is pre-trained.
- Just the feature extraction layers are used.
- We can adjust weights `content_weight` and `style_weight` for better results.

### How to Run
- Add your photo in `in` folder
- Add your style in `in` folder
- `python3 main.py`

### Results
Photo:

<img src="https://github.com/vbatistella/CNN-style-transfer/blob/main/in/in_photo.jpg" width="288" height="512">

Style:

<img src="https://github.com/vbatistella/CNN-style-transfer/blob/main/in/in_style.png" width="288" height="512">

Result:

<img src="https://github.com/vbatistella/CNN-style-transfer/blob/main/out/out.png" width="288" height="512">