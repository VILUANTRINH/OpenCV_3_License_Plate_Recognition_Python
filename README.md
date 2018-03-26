## Presentation
This project is based on License Plate Recognition on this github:
https://github.com/MicrocontrollersAndMore/OpenCV_3_License_Plate_Recognition_Python

The following video explains general process:
https://www.youtube.com/watch?v=fJcl6Gw1D8k

What has been added:
- Specific model to recognize Vietnam's motorbike plate
- Use CNN model instead of KNN model
- Add more constraints to increase results

You also can see DocsAndPresentation folder

## Instruction
To recognize character in an image
```
python Main.py detect ./test/train_data/plate_2.jpg --steps True --save True
```

`--save=True` save image with recognized chars in outputs
`--steps=True` will show step by step recognition process

## Self Exploration
To generate chars from a font. All fonts contained in `fonts` folder will be converted to template
```
python GenChars.py --font True
```

To get trained chars from real plate
```
python Main.py gen TRAIN_PLATE_FOLDER REAL_CHARS_FOLDER
```

To augment chars from real chars. Check required folders in `GenChars.py`
```
python GenChars.py --gen_char 10
```

If you want to retrain model by yourself, you can check `Train_CNN_Model.ipynb`
