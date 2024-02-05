# Solve captcha

#### Short description:

This project predicts randomly generated captchas from ```https://captchamaker.com/```

#### How it works?

First you need to parse the dataset of letters for recognition, see ```parse_captchas/parse.py```.
The parsed letters will be stored in `extracted_letters`, preserving the markup.

The ```selenium``` library is used for parsing

Let's prepare the training data and use libraries
```cv2```,
```joblib```,
```numpy```,
```sklearn```,
see ```solve_captchas/utils```

Then we will create the ```Sequential``` model from the ```Keras``` library, train it on preprocessed data, save
model ```sequential.keras```,
see: ```solve_captchas/train_model.py```

Get result of captcha prediction, see `solve_captchas\predict_captha.py`
The parsed captchas will be stored in `extracted_capthas`, preserving unique name.

