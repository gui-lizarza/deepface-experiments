import os
from tqdm import tqdm
import pandas as pd

from deepface.basemodels import VGGFace, OpenFace, Facenet, FbDeepFace, DeepID
from deepface.commons import functions, realtime, distance as dst

def make_dict(dataset_path):

    dic = {}
    os.chdir(dataset_path)
    class_folders = os.listdir('.')

    for class_folder in class_folders:

        os.chdir(class_folder)
        images = os.listdir('.')

        i = 1
        for image in images:
            os.rename(image, class_folder + '.{}'.format(i) + '.jpg')
            dic['{}.{}'.format(class_folder, i)] = dataset_path + chr(92) + class_folder + chr(92) + image
            i = i + 1

        os.chdir('..')

    os.chdir('..')

    return dic

def save_hash(img_dict, model_name='Ensemble', model=None, enforce_detection=True, detector_backend = 'mtcnn'):

    img_list = list(img_dict.keys())

    if model_name == 'Ensemble':

        print("Ensemble learning enabled")
        model_names = ["VGG-Face", "Facenet", "OpenFace", "DeepFace"]

        if model == None:

            model = {}
            model_pbar = tqdm(range(0, 4), desc='Loading face recognition models')
            
            for index in model_pbar:
                
                if index == 0:
                    model_pbar.set_description("Loading VGG-Face")
                    model["VGG-Face"] = VGGFace.loadModel()
                elif index == 1:
                    model_pbar.set_description("Loading Google FaceNet")
                    model["Facenet"] = Facenet.loadModel()
                elif index == 2:
                    model_pbar.set_description("Loading OpenFace")
                    model["OpenFace"] = OpenFace.loadModel()
                elif index == 3:
                    model_pbar.set_description("Loading Facebook DeepFace")
                    model["DeepFace"] = FbDeepFace.loadModel()

        pbar = tqdm(range(0,len(img_list)), desc='Avaliando Hash')
        df = pd.DataFrame(columns = model_names, index = list(img_dict.keys()))
        df = df.astype(object)
        representation = []

        for index in pbar:

            erro = False
            img1_key = img_list[index]
            representation = []
            
            for i in  model_names:
                custom_model = model[i]

                input_shape = functions.find_input_shape(custom_model)	
                input_shape_x = input_shape[0]; input_shape_y = input_shape[1]
                
                try:
                    img1 = functions.preprocess_face(img=img_dict[img1_key]
                    , target_size=(input_shape_y, input_shape_x)
                    , enforce_detection = enforce_detection
                    , detector_backend = detector_backend)
                except:
                    erro = True
                    break

                img1_representation = custom_model.predict(img1)[0,:]
                representation.append(img1_representation)

            if erro:
                continue
            
            df.loc[img_list[index]] = representation				

            return df

    print('Por enquanto, aceitamos apenas Ensemble e nossos modelos predefinidos.')
    return None


dir = make_dict('Turing_Faces')
df = save_hash(dir)
print(df.to_string())