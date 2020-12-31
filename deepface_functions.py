import os
from tqdm import tqdm
import pandas as pd
import numpy as np

from deepface.basemodels import VGGFace, OpenFace, Facenet, FbDeepFace, Boosting
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
            if not os.path.isfile(class_folder + '{}'.format(i) + '.jpg'):
              os.rename(image, class_folder + '{}'.format(i) + '.jpg')
            i = i + 1

        images = os.listdir('.')
        
        i = 1
        for image in images:
            dic['{}{}'.format(class_folder, i)] = os.path.join(dataset_path, class_folder, image)
            i = i + 1

        os.chdir('..')

    os.chdir('..')

    return dic

def save_hash(img_dict, model_name='Ensemble', enforce_detection=True, detector_backend = 'mtcnn'):

    img_list = list(img_dict.keys())

    model = {}

    if model_name == 'Ensemble':

        print("Ensemble learning enabled")
        model_names = ["VGG-Face", "Facenet", "OpenFace", "DeepFace"]

        model_pbar = tqdm(range(4), desc='Loading face recognition models', disable = False)
        
        for index in model_pbar:
            
            if index == 0:
                model_pbar.set_description("Loading VGG-Face...")
                model["VGG-Face"] = VGGFace.loadModel()
            elif index == 1:
                model_pbar.set_description("Loading Google FaceNet...")
                model["Facenet"] = Facenet.loadModel()
            elif index == 2:
                model_pbar.set_description("Loading OpenFace...")
                model["OpenFace"] = OpenFace.loadModel()
            elif index == 3:
                model_pbar.set_description("Loading Facebook DeepFace...")
                model["DeepFace"] = FbDeepFace.loadModel()

    else:
  
        model_names = []; model_names.append(model_name)
        
        if model_name == 'VGG-Face':
            print(model_name + " enabled")
            print("Loading VGG-Face...")
            model["VGG-Face"] = VGGFace.loadModel()
        elif model_name == 'FaceNet':
            print(model_name + " enabled")
            print("Loading Google FaceNet...")
            model["Facenet"] = Facenet.loadModel()
        elif model_name == 'OpenFace':
            print(model_name + " enabled")
            print("Loading OpenFace...")
            model["OpenFace"] = OpenFace.loadModel()
        elif model_name == 'DeepFace':
            print(model_name + " enabled")
            print("Loading Facebook DeepFace...")
            model["DeepFace"] = FbDeepFace.loadModel()
        else:
            raise NameError ('Invalid model_name: ' + model_name)

    disable_option = False if len(img_list) > 1 else True
    pbar = tqdm(range(len(img_list)), desc='Avaliando Hash', disable = disable_option)

    df = pd.DataFrame(columns = model_names, index = list(img_dict.keys()))
    df = df.astype(object)
    representation = []
    erro_keys = []

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
                , enforce_detection=enforce_detection
                , detector_backend=detector_backend)
            except:
                print('Erro na representação!')
                erro = True
                erro_keys.append(img1_key)
                break

            img1_representation = custom_model.predict(img1)[0,:]
            representation.append(img1_representation)

        if erro:
            continue
        
        df.loc[img_list[index]] = representation	

    for i in erro_keys:
        df.drop(index = i, inplace = True)				

    return df

def metrics_dataframe(img_df, chosen_metric='Ensemble', one_compair=False):

    if chosen_metric=='Ensemble':
        metrics = ['Cosine', 'Euclidean', 'L2']
    elif chosen_metric=='Cosine' or chosen_metric=='Euclidean' or chosen_metric=='L2':
        metrics=[]; metrics.append(chosen_metric)
    else:
        raise NameError ('Invalid chosen_metric: ' + chosen_metric)

    model = list(img_df.columns)
    column_names = []
    index_names = []
    for i in model:
        for j in metrics:
            column_names.append(j + ' with ' + i)

    for i in img_df.index:
        for j in img_df.index:
            index_names.append(i + ' - ' + j)

    metrics_df = pd.DataFrame(columns=column_names, index=index_names)
    metrics_df = metrics_df.astype(object)

    if one_compair:

        image_index = img_df.index[0]
        people_pbar = tqdm(range(1, len(img_df.index)), desc = 'Avaliando a primeira pessoa em relação a todas as outras.')
        for i in people_pbar:

            real_index = img_df.index[i]

            distances = []
            for model in img_df.columns:

                if 'Cosine' in metrics:
                    cosine_distance = dst.findCosineDistance(img_df.at[real_index, model], img_df.at[image_index, model])
                    distances.append(cosine_distance)
                if 'Euclidean' in metrics:
                    euclidean_distance = dst.findEuclideanDistance(img_df.at[real_index, model], img_df.at[image_index, model])
                    distances.append(euclidean_distance)
                if 'L2' in metrics:
                    l2_distance = dst.findEuclideanDistance(dst.l2_normalize(img_df.at[real_index, model]), dst.l2_normalize(img_df.at[image_index, model]))
                    distances.append(l2_distance)

            string = image_index + '-' + real_index
            metrics_df.loc[string] = distances

    else:

        people_pbar = tqdm(range(len(img_df.index)), desc = 'Avaliando cada pessoa em relação a todas as outras.')
        for i in people_pbar:
			
            real_index = img_df.index[i]

            for j in people_pbar:

                real_jndex = img_df.index[j]


                distances = []	
                for model in img_df.columns:
				
                    if 'Cosine' in metrics:
                        cosine_distance = dst.findCosineDistance(img_df.at[real_index, model], img_df.at[real_jndex, model])
                        distances.append(cosine_distance)
                    if 'Euclidean' in metrics:
                        euclidean_distance = dst.findEuclideanDistance(img_df.at[real_index, model], img_df.at[real_jndex, model])
                        distances.append(euclidean_distance)
                    if 'L2' in metrics:
                        l2_distance = dst.findEuclideanDistance(dst.l2_normalize(img_df.at[real_index, model]), dst.l2_normalize(img_df.at[real_jndex, model]))
                        distances.append(l2_distance)

                string = real_index + '-' + real_jndex
                metrics_df.loc[string] = distances

    metrics_df.dropna(inplace = True)

    return metrics_df

def ensemble_dataframe(metrics_df):

    columns_names = ['Verified', 'Ground Truth', 'Score']
    metrics_df = metrics_df.drop('Euclidean with OpenFace', axis = 1)
    verification_df = pd.DataFrame(columns = columns_names, index = metrics_df.index)

    deepface_ensemble = Boosting.build_gbm()

    comp_pbar = tqdm(range(len(metrics_df.index)), desc = 'Realizando o ensemble para cada par de indivíduos.')
    for i in comp_pbar:

        index = metrics_df.index[i]
        prediction = deepface_ensemble.predict(np.expand_dims(np.array(metrics_df.loc[index]), axis=0))[0]

        verified = np.argmax(prediction) == 1
        if verified: identified = "true"
        else: identified = "false"

        score = prediction[np.argmax(prediction)]

        names = index.split('-')
        if (names[0])[:-1] == (names[1])[:-1]:
            ground_truth = "true"
        else: ground_truth = "false"

        ensembles = [identified, ground_truth, score]
        verification_df.loc[index] = ensembles

    return verification_df

def f1_calculation(verif_df):

    TP, FP, TN, FN = 0, 0, 0, 0   
    column_names = ['Accuracy', 'Recall', 'Precision', 'F1']
    for index in verif_df.index:

        if (verif_df.loc[index][0] == verif_df.loc[index][1]) and (verif_df.loc[index][0] == 'true'):
            TP += 1
        elif (verif_df.loc[index][0] != verif_df.loc[index][1]) and (verif_df.loc[index][0] == 'true'):
            FP += 1 
        elif (verif_df.loc[index][0] == verif_df.loc[index][1]) and (verif_df.loc[index][0] != 'true'):
            TN += 1
        else:
            FN += 1

    accuracy = (TP+TN)/(TP+TN+FP+FN)
    recall = TN/(TN+FN)
    precision = TP/(TP+FP)
    f1 = 2*(recall*precision)/(recall+precision)
    f1_df = pd.DataFrame([list([accuracy, recall, precision, f1])], columns = column_names, index = ['Results:'])

    return f1_df





            

