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

![photo](in/in_photo.jpg =288x512)

Style:

![style](in/in_style.png =288x512)

Result:

![result](out/out.png =288x512)