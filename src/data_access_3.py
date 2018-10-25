
import tensorflow as tf

filenames = ["/home/benmackenzie/Projects/ML/export/heart-0.csv"]


feature_names = ["max_heart_rate","age","sex","chest_pain_type","resting_blood_pressure",
                 "cholesterol","fasting_blood_sugar","resting_ecg","exercise_angina","oldpeak",
                 "slope","colored_vessels","thal","datetime","postalcode","narrowing_diagnosis"]



def prep(*features):
 f = dict(zip(feature_names, features))
 label = f['narrowing_diagnosis']
 f.pop('narrowing_diagnosis')
 return (f, label)

#dataset = tf.contrib.data.CsvDataset(filenames, [[153],[63], ['male'], ['typical angina'], [145], [233], [0], ['left ventricular hypertrophy'],
# ['no'], [2.3], ['downsloping'], [0], ['fixed defect'],['2018-08-07'], ['92129'], [0]], header=True)

#dataset = dataset.batch(32)
#dataset = dataset.map(prep)
#iterator = dataset.make_one_shot_iterator()
#next_element = iterator.get_next()
#sess = tf.Session()
#print(sess.run(next_element))




def csv_input_function():
    dataset = tf.contrib.data.CsvDataset(filenames, [[153], [63], ['male'], ['typical angina'], [145], [233], [0],
                                                  ['left ventricular hypertrophy'],
                                                  ['no'], [2.3], ['downsloping'], [0], ['fixed defect'], ['2018-08-07'],
                                                  ['92129'], [0]], header=True)

    dataset = dataset.batch(32)
    dataset = dataset.map(prep)
    return dataset


