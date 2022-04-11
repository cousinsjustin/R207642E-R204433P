from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
# import split_video as sv
from os import listdir
from os.path import isfile, join


def classifyImages():
    model = VGG16()
    from keras.applications.vgg16 import decode_predictions
    classify = []
    frames = [ join('.\\frames', f) for f in listdir('.\\frames') if isfile(join('.\\frames', f)) ]
    for i in range(len(frames)):    

        image = load_img(frames[i], target_size=(224, 224)) 
        image = img_to_array(image)

        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)

        img_pred = model.predict(image)
        label = decode_predictions(img_pred) 

        label = label[0][0]
        result =  label[1]

        classify.append(result)
    print(classify)
    
    return classify
if __name__ == "__main__":
    classifyImages()